
#include <sstream>
#include <iomanip>
#include <cstdio>
#include <cmath>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "RenderData.cuh"

#include "Utils.cuh"
#include "ThinLensCamera.cuh"

// Note: arguments MUST be by value. Make sure they are fast to copy
__global__ void RenderKernel( const rt::RenderData renderData )
{
  const math::uvec2 pixel( blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y );
  if ( pixel.x >= renderData.mDimensions.x || pixel.y >= renderData.mDimensions.y )
  {
    return;
  }

  // TODO: get n samples instead of one and use their average color
  const rt::Ray ray( renderData.mCamera.GetRay( pixel, renderData.mDimensions ) );

  // save final pixel color
  const uint32_t offset( pixel.x + pixel.y * renderData.mDimensions.x );
  renderData.mPixelBuffer[offset] = utils::Color( 255 * ( glm::abs( ray.direction().x ) )
                                                  , 255 * ( glm::abs( ray.direction().y ) )
                                                  , 255 * ( glm::abs( ray.direction().z ) ) );

  //renderData.mPixelBuffer[0] = 0xFFFFFFFF;
}

cudaError_t RunRenderKernel( rt::RenderData& renderData )
{
  // TODO fast way, do better!
  const dim3 threadsPerBlock( 32, 32, 1 );
  const dim3 blocksPerGrid( static_cast<uint32_t>( glm::ceil( renderData.mDimensions.x / static_cast<float>( threadsPerBlock.x ) ) )
                            , static_cast<uint32_t>( glm::ceil( renderData.mDimensions.y / static_cast<float>( threadsPerBlock.y ) ) )
                            , 1 );

  cudaEvent_t start, stop;
  cudaEventCreate( &start );
  cudaEventCreate( &stop );

  cudaEventRecord( start, 0 );
  RenderKernel<<<blocksPerGrid, threadsPerBlock>>> ( renderData );
  cudaEventRecord( stop, 0 );
  cudaEventSynchronize( stop );

  float time = 0.0f;
  cudaEventElapsedTime( &time, start, stop );
  return cudaGetLastError();
}
