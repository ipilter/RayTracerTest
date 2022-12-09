#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "RayTracerImpl.cuh"
#include "Random.cuh"
#include "DeviceUtils.cuh"
#include "RenderKernel.cuh"

namespace rt
{

RayTracerImpl::RayTracerImpl( const math::uvec2& pixelBufferSize )
  : mPixelBufferSize( pixelBufferSize )
  , mRandomStates( nullptr )
{
  random::RunInitRandom( pixelBufferSize, mRandomStates );
}

RayTracerImpl::~RayTracerImpl()
{
  const cudaError_t err = cudaFree( mRandomStates );
  if ( err != cudaSuccess )
  {
    // TODO handle error
    const auto str = cudaGetErrorString( err );
  }
}

void RayTracerImpl::Trace( rt::color_t* pixelBufferPtr
            , const uint32_t sampleCount
            , const float fov
            , const float focalLength
            , const float aperture )
{
  rt::ThinLensCamera camera( math::vec3( 0.0f, 0.0f, 0.0f )
                             , math::vec3( 0.0f, 0.0f, -1.0f )
                             , math::vec3( 0.0f, 1.0f, 0.0f )
                             , fov, focalLength, aperture );

  cudaError_t err = RunRenderKernel( pixelBufferPtr, mPixelBufferSize, camera, sampleCount, mRandomStates );
  if ( err != cudaSuccess )
  {
    throw std::runtime_error( std::string( "RunRenderKernel failed: " ) + cudaGetErrorString( err ) );
  }
}

void RayTracerImpl::Resize( const math::uvec2& size )
{
  mPixelBufferSize = size;
  const cudaError_t err = cudaFree( mRandomStates );
  if ( err == cudaSuccess )
  {
    random::RunInitRandom( size, mRandomStates );
  }
  else
  {
    // TODO handle error
    const auto str = cudaGetErrorString( err );
  }
}

cudaError_t RayTracerImpl::RunRenderKernel( rt::color_t* pixelBufferPtr
                                            , const math::uvec2& pixelBufferSize
                                            , rt::ThinLensCamera& camera
                                            , const uint32_t sampleCount
                                            , curandState_t* randomStates )
{
  // TODO fast way, do better!
  const dim3 threadsPerBlock( 32, 32, 1 );
  const dim3 blocksPerGrid( static_cast<uint32_t>( glm::ceil( pixelBufferSize.x / static_cast<float>( threadsPerBlock.x ) ) )
                            , static_cast<uint32_t>( glm::ceil( pixelBufferSize.y / static_cast<float>( threadsPerBlock.y ) ) )
                            , 1 );

  cudaEvent_t start, stop;
  cudaEventCreate( &start );
  cudaEventCreate( &stop );

  cudaEventRecord( start, 0 );
  RenderKernel<<<blocksPerGrid, threadsPerBlock>>> ( pixelBufferPtr, pixelBufferSize, camera, sampleCount, randomStates );

  // this blocks the CPU till all the GPU commands are executed (kernel, copy, etc)
  //cudaDeviceSynchronize();

  cudaEventRecord( stop, 0 );
  cudaEventSynchronize( stop );

  float time = 0.0f;
  cudaEventElapsedTime( &time, start, stop );
  return cudaGetLastError();
}

}