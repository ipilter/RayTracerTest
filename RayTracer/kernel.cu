
#include <sstream>
#include <iomanip>
#include <cstdio>
#include <cmath>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "RenderData.h"

#include "Utils.cuh"

__global__ void UpdateTextureKernel( uint32_t* rgba, int const width, int const height )
{
  const uint32_t x( blockIdx.x * blockDim.x + threadIdx.x );
  const uint32_t y( blockIdx.y * blockDim.y + threadIdx.y );
  if ( x >= width || y >= height )
  {
    return;
  }

  const uint32_t offset = x + y * width;
  rgba[offset] = utils::Color( 255 * ( x / static_cast<float>( width ) )
                               , 255 * ( y / static_cast<float>( height ) )
                               , 0 );
}

cudaError_t RunUpdateTextureKernel( rt::RenderData& renderData )
{
  // TODO fast way, do better!
  dim3 threadsPerBlock( 32, 32, 1 );
  dim3 blocksPerGrid( static_cast<uint32_t>( glm::ceil( renderData.Dimensions().x / static_cast<float>( threadsPerBlock.x ) ) )
                      , static_cast<uint32_t>( glm::ceil( renderData.Dimensions().y / static_cast<float>( threadsPerBlock.y ) ) )
                      , 1 );

  cudaEvent_t start, stop;
  cudaEventCreate( &start );
  cudaEventCreate( &stop );

  cudaEventRecord( start, 0 );
  UpdateTextureKernel<<<blocksPerGrid, threadsPerBlock>>> ( renderData.PixelBuffer(), renderData.Dimensions().x, renderData.Dimensions().y );
  cudaEventRecord( stop, 0 );
  cudaEventSynchronize( stop );

  float time = 0.0f;
  cudaEventElapsedTime( &time, start, stop );
  return cudaGetLastError();
}
