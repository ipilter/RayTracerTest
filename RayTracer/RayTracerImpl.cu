#include <cuda_runtime.h>
#include <chrono>
#include <device_launch_parameters.h>

#include "RayTracerImpl.cuh"
#include "Random.cuh"
#include "DeviceUtils.cuh"
#include "Kernels.cuh"
#include "RenderBuffer.cuh"
#include "RaytracerCallback.h"

#include "Common\Logger.h"

namespace rt
{

RayTracerImpl::RayTracerImpl( const math::uvec2& imageSize
                              , const math::vec3& cameraPosition
                              , const math::vec2& cameraAngles
                              , const float fov
                              , const float focalLength
                              , const float aperture )
  : mBufferSize( imageSize )
  , mRenderBuffer( nullptr )
  , mSampleCountBuffer( nullptr )
  , mImageBuffer( nullptr )
  , mRandomStates( nullptr )
  , mCamera( new rt::ThinLensCamera( cameraPosition, cameraAngles, fov, focalLength, aperture ) )
  , mStopped( false )
{
  try
  {
    random::CreateStates( mBufferSize, mRandomStates );
    rt::CreateRenderBuffer( mBufferSize, ChannelCount(), mRenderBuffer );
    rt::CreateSampleCountBuffer( mBufferSize, mSampleCountBuffer );
    rt::CreateImageBuffer( mBufferSize, mImageBuffer );

    logger::Logger::Instance() << "Raytracer created. Image buffer size: " << mBufferSize << "\n";
  }
  catch ( const std::exception& e )
  {
    logger::Logger::Instance() << "RayTracerImpl construction failed. Reason: " << e.what() << "\n";
  }
}

RayTracerImpl::~RayTracerImpl()
{
  // If rendering task is in progress, cancel it
  mStopped = true;
  if ( mThread.joinable() )
  {
    mThread.join();
  }

  // TODO free temporary image buffer - storage of converted image pixels (RGBA)
  ReleaseBuffers();
}

void RayTracerImpl::Trace( const uint32_t iterationCount
                           , const uint32_t samplesPerIteration
                           , const uint32_t updateInterval )
{
  // cancel previous operation, if any
  if ( mThread.joinable() )
  {
    mStopped = true;
    mThread.join();
    mStopped = false;
  }

  // Run rendering function async. TODO use pool instead of creating a new thread
  mThread = std::thread( std::bind( &RayTracerImpl::TraceFunct
                                    , this
                                    , iterationCount
                                    , samplesPerIteration
                                    , updateInterval ) );
}

void RayTracerImpl::Stop()
{
  mStopped = true;
}

void RayTracerImpl::Resize( const math::uvec2& size )
{
  ReleaseBuffers();

  mBufferSize = size;
  random::CreateStates( mBufferSize, mRandomStates );
  rt::CreateRenderBuffer( mBufferSize, ChannelCount(), mRenderBuffer );
  rt::CreateSampleCountBuffer( mBufferSize, mSampleCountBuffer );
  rt::CreateImageBuffer( mBufferSize, mImageBuffer );
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

void RayTracerImpl::SetUpdateCallback( rt::CallBackFunction callback )
{
  mUpdateCallback = callback;
}

void RayTracerImpl::SetFinishedCallback( rt::CallBackFunction callback )
{
  mFinishedCallback = callback;
}

cudaError_t RayTracerImpl::RunConverterKernel()
{
  const dim3 threadsPerBlock( 32, 32, 1 );
  const dim3 blocksPerGrid( static_cast<uint32_t>( glm::ceil( mBufferSize.x / static_cast<float>( threadsPerBlock.x ) ) )
                            , static_cast<uint32_t>( glm::ceil( mBufferSize.y / static_cast<float>( threadsPerBlock.y ) ) )
                            , 1 );

  ConverterKernel<<<blocksPerGrid, threadsPerBlock>>>( mBufferSize
                                                       , ChannelCount()
                                                       , mRenderBuffer
                                                       , mSampleCountBuffer
                                                       , mImageBuffer );
  return cudaGetLastError();
}

cudaError_t RayTracerImpl::RunTraceKernel( const uint32_t sampleCount )
{
  // TODO fast way, do better!
  const dim3 threadsPerBlock( 32, 32, 1 );
  const dim3 blocksPerGrid( static_cast<uint32_t>( glm::ceil( mBufferSize.x / static_cast<float>( threadsPerBlock.x ) ) )
                            , static_cast<uint32_t>( glm::ceil( mBufferSize.y / static_cast<float>( threadsPerBlock.y ) ) )
                            , 1 );

  cudaEvent_t start, stop;
  cudaEventCreate( &start );
  cudaEventCreate( &stop );

  cudaEventRecord( start, 0 );
  TraceKernel<<<blocksPerGrid, threadsPerBlock>>> ( mRenderBuffer
                                                    , mSampleCountBuffer
                                                    , mBufferSize
                                                    , ChannelCount()
                                                    , *mCamera
                                                    , sampleCount
                                                    , mRandomStates );

  cudaEventRecord( stop, 0 );
  cudaEventSynchronize( stop ); // TODO: make this switchable (on/off)

  float time = 0.0f;
  cudaEventElapsedTime( &time, start, stop );

  return cudaGetLastError();
}

__host__ void RayTracerImpl::TraceFunct( const uint32_t iterationCount
                                         , const uint32_t samplesPerIteration
                                         , const uint32_t updateInterval )
{
  try
  {
    rt::ClearRenderBuffer( mBufferSize, ChannelCount(), mRenderBuffer );
    rt::ClearSampleCountBuffer( mBufferSize, mSampleCountBuffer );

    cudaError_t err = cudaSuccess;
    for ( uint32_t i( 0 ); !mStopped && i < iterationCount; ++i )
    {
      // TODO: make kernel call cancellable if possible (imageine long runtimer here, cancel operation would wait for this call)
      err = RunTraceKernel( samplesPerIteration );
      if ( err != cudaSuccess )
      {
        throw std::runtime_error( std::string( "RunTraceKernel failed: " ) + cudaGetErrorString( err ) );
      }

      // check if update is needed
      if ( mUpdateCallback != nullptr && i > 0 && updateInterval > 0 && i % updateInterval == 0 )
      {
        // wait for the scheduled commands to be executed
        cudaDeviceSynchronize();

        // run render -> image conversion
        err = RunConverterKernel();
        if ( err != cudaSuccess )
        {
          throw std::runtime_error( std::string( "RunConverterKernel failed: " ) + cudaGetErrorString( err ) );
        }

        // notify view to update the view's texture
        mUpdateCallback( mImageBuffer, mBufferSize.x * mBufferSize.y * sizeof( rt::Color ) );
      }

      using namespace std::chrono_literals;
      //std::this_thread::sleep_for( 500ms );
    }

    // early reaturn if cancel was called on us
    if ( mStopped )
    {
      // TODO: any UI update?
      return;
    }

    // wait for the scheduled commands to be executed
    cudaDeviceSynchronize();

    // run render -> image conversion
    err = RunConverterKernel();
    if ( err != cudaSuccess )
    {
      throw std::runtime_error( std::string( "RunConverterKernel failed: " ) + cudaGetErrorString( err ) );
    }

    // notify view that we are done
    if ( mFinishedCallback != nullptr )
    {
      mFinishedCallback( mImageBuffer, mBufferSize.x * mBufferSize.y * sizeof( rt::Color ) );
    }
  }
  catch ( const std::exception& /*e*/ )
  {
    // TODO: error handling
  }
  catch ( ... )
  {
    // TODO: error handling
  }
}

void RayTracerImpl::ReleaseBuffers()
{
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

  err = cudaFree( mSampleCountBuffer );
  if ( err != cudaSuccess )
  {
    logger::Logger::Instance() << "Error: cudaFree failed freeing mSampleCountBuffer. (" << cudaGetErrorString( err ) << "\n";
  }

  err = cudaFree( mImageBuffer );
  if ( err != cudaSuccess )
  {
    logger::Logger::Instance() << "Error: cudaFree failed freeing mImageBuffer. (" << cudaGetErrorString( err ) << "\n";
  }
}

}
