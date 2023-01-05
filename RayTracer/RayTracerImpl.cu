#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "RayTracerImpl.cuh"
#include "Random.cuh"
#include "DeviceUtils.cuh"
#include "RenderKernel.cuh"
#include "Render.cuh"

#include "Common\Logger.h"

namespace rt
{

RayTracerImpl::RayTracerImpl( const math::uvec2& pixelBufferSize
                              , const math::vec3& cameraPosition
                              , const math::vec2& cameraAngles
                              , const float fov
                              , const float focalLength
                              , const float aperture )
  : mPixelBufferSize( pixelBufferSize )
  , mRandomStates( nullptr )
  , mRenderBuffer( nullptr )
  , mCamera( new rt::ThinLensCamera( cameraPosition, cameraAngles, fov, focalLength, aperture ) )
  , mCancelled( false )
  , mThread()
{
  const uint32_t channelCount = 4;

  random::CreateStates( mPixelBufferSize, mRandomStates );
  render::CreateRenderBuffer( mPixelBufferSize, channelCount, mRenderBuffer );
  render::ClearRenderBuffer( mPixelBufferSize, channelCount, mRenderBuffer );

  logger::Logger::Instance() << "Raytracer created. Pixel buffer size: " << mPixelBufferSize << "\n";
}

RayTracerImpl::~RayTracerImpl()
{
  // If rendering task is in progress, cancel it
  mCancelled = true;
  if ( mThread.joinable() )
  {
    mThread.join();
  }

  cudaError_t err = cudaFree( mRandomStates );
  if ( err != cudaSuccess )
  {
    logger::Logger::Instance() << "Error: cudaFree failed freeing mRandomStates. (" << cudaGetErrorString( err ) << "\n";
  }

  err = cudaFree( mRenderBuffer );
  if ( err != cudaSuccess )
  {
    logger::Logger::Instance() << "Error: cudaFree failed freeing mRenderBuffer. (" << cudaGetErrorString( err ) << "\n";
  }
}

void RayTracerImpl::Trace( rt::color_t* pixelBufferPtr
                           , const uint32_t iterationCount
                           , const uint32_t samplesPerIteration
                           , const uint32_t updatesOnIteration )
{
  // cancel previous operation, if any
  mCancelled = true;
  if ( mThread.joinable() )
  {
    mThread.join();
  }
  mCancelled = false;

  // Run rendering function async
  mThread = std::thread( std::bind( &RayTracerImpl::TraceFunct
                                    , this
                                    , pixelBufferPtr
                                    , iterationCount
                                    , samplesPerIteration
                                    , updatesOnIteration ) );
}

void RayTracerImpl::Cancel()
{
  mCancelled = true;
}

void RayTracerImpl::Resize( const math::uvec2& size )
{
  mPixelBufferSize = size;
  cudaError_t err = cudaFree( mRandomStates );
  if ( err != cudaSuccess )
  {
    logger::Logger::Instance() << "Error: cudaFree failed freeing mRandomStates. (" << cudaGetErrorString( err ) << "\n";
  }

  err = cudaFree( mRenderBuffer );
  if ( err != cudaSuccess )
  {
    logger::Logger::Instance() << "Error: cudaFree failed freeing mRenderBuffer. (" << cudaGetErrorString( err ) << "\n";
  }

  random::CreateStates( mPixelBufferSize, mRandomStates );
  render::CreateRenderBuffer( mPixelBufferSize, 4, mRenderBuffer );
}

void RayTracerImpl::SetCameraParameters( const float fov
                                         , const float focalLength
                                         , const float aperture )
{
  mCamera->Fov( fov );
  mCamera->FocalLength( focalLength );
  mCamera->Aperture( aperture );
}

void RayTracerImpl::RotateCamera( const math::vec2& angles )
{
  mCamera->Rotate( angles );
}

void RayTracerImpl::SetUpdateCallback( CallBackFunction callback )
{
  mUpdateCallback = callback;
}

void RayTracerImpl::SetFinishedCallback( CallBackFunction callback )
{
  mFinishedCallback = callback;
}

cudaError_t RayTracerImpl::RunConverterKernel( const math::uvec2& bufferSize
                                                      , const uint32_t channelCount
                                                      , float*& renderBuffer
                                                      , rt::color_t* pixelBufferPtr )
{
  const dim3 threadsPerBlock( 32, 32, 1 );
  const dim3 blocksPerGrid( static_cast<uint32_t>( glm::ceil( bufferSize.x / static_cast<float>( threadsPerBlock.x ) ) )
                            , static_cast<uint32_t>( glm::ceil( bufferSize.y / static_cast<float>( threadsPerBlock.y ) ) )
                            , 1 );

  ConverterKernel<<<blocksPerGrid, threadsPerBlock>>>( bufferSize, channelCount, renderBuffer, pixelBufferPtr );
  return cudaGetLastError();
}

cudaError_t RayTracerImpl::RunRenderKernel( float* renderBuffer
                                            , const math::uvec2& bufferSize
                                            , const uint32_t channelCount
                                            , rt::ThinLensCamera& camera
                                            , const uint32_t sampleCount
                                            , curandState_t* randomStates )
{
  // TODO fast way, do better!
  const dim3 threadsPerBlock( 32, 32, 1 );
  const dim3 blocksPerGrid( static_cast<uint32_t>( glm::ceil( bufferSize.x / static_cast<float>( threadsPerBlock.x ) ) )
                            , static_cast<uint32_t>( glm::ceil( bufferSize.y / static_cast<float>( threadsPerBlock.y ) ) )
                            , 1 );

  cudaEvent_t start, stop;
  cudaEventCreate( &start );
  cudaEventCreate( &stop );

  cudaEventRecord( start, 0 );
  RenderKernel<<<blocksPerGrid, threadsPerBlock>>> ( renderBuffer, bufferSize, channelCount, camera, sampleCount, randomStates );

  cudaEventRecord( stop, 0 );
  cudaEventSynchronize( stop ); // TODO: make this switchable (on/off)

  //cudaDeviceSynchronize(); // this blocks the CPU till all the GPU commands are executed (kernel, copy, etc)

  float time = 0.0f;
  cudaEventElapsedTime( &time, start, stop );

  return cudaGetLastError();
}

__host__ void RayTracerImpl::TraceFunct( rt::color_t* pixelBufferPtr
                                         , const uint32_t iterationCount
                                         , const uint32_t samplesPerIteration
                                         , const uint32_t updatesOnIteration )
{
  try
  {
    const uint32_t channelCount = 4;

    render::ClearRenderBuffer( mPixelBufferSize, channelCount, mRenderBuffer );

    cudaError_t err = cudaSuccess;
    for ( auto i = 0u; i < iterationCount; ++i ) // TODO && !mIsCancelled
    {
      RunRenderKernel( mRenderBuffer, mPixelBufferSize, channelCount, *mCamera, samplesPerIteration, mRandomStates );
      if ( err != cudaSuccess )
      {
        throw std::runtime_error( std::string( "RunRenderKernel failed: " ) + cudaGetErrorString( err ) );
      }

      if ( mUpdateCallback != nullptr && updatesOnIteration > 0 && i % updatesOnIteration == 0 )
      {
        err = RunConverterKernel( mPixelBufferSize, channelCount, mRenderBuffer, pixelBufferPtr );
        if ( err != cudaSuccess )
        {
          throw std::runtime_error( std::string( "RunConverterKernel failed: " ) + cudaGetErrorString( err ) );
        }

        // notify view to update the view's texture
        mUpdateCallback();
      }
    }

    // convert render buffer to image and store it in PBO
    err = RunConverterKernel( mPixelBufferSize, channelCount, mRenderBuffer, pixelBufferPtr );
    if ( err != cudaSuccess )
    {
      throw std::runtime_error( std::string( "RunConverterKernel failed: " ) + cudaGetErrorString( err ) );
    }

    // notify view that we are done
    if ( mFinishedCallback != nullptr )
    {
      mFinishedCallback();
    }
  }
  catch ( const std::exception& e )
  {
    // TODO: error handling
  }
}

}
