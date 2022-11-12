
#include <sstream>
#include <iomanip>
#include <stdio.h>
#include <cmath>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "RenderData.h"

namespace utils
{
__device__ uint8_t Component( const uint32_t& color, const uint32_t& idx )
{
  switch ( idx )
  {
    case 0:
      return static_cast<uint8_t>( ( color & 0x000000FF ) >> 0 );
    case 1:
      return static_cast<uint8_t>( ( color & 0x0000FF00 ) >> 8 );
    case 2:
      return static_cast<uint8_t>( ( color & 0x00FF0000 ) >> 16 );
    case 3:
      return static_cast<uint8_t>( ( color & 0xFF000000 ) >> 24 );
    default:
      return 0;
  }
}

__device__ uint32_t Color( const uint8_t r = 0, const uint8_t g = 0, const uint8_t b = 0, const uint8_t a = 255 )
{
  return ( r << 0 ) | ( g << 8 ) | ( b << 16 ) | ( a << 24 );
}

} // ::utils

__global__ void UpdateTextureKernel( uint32_t* rgba, int const width, int const height )
{
  const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

  if ( x >= width || y >= height )
  {
    return;
  }

  const unsigned int offset = x + y * width;
  rgba[offset] = utils::Color( 255, 255, 255 );
                 //utils::Color( 255 * ( x / static_cast<double>( width ) )
                 //              , 255 * ( y / static_cast<double>( height ) )
                 //              , 0 );
  rgba[0] = 42;
}

cudaError_t RunUpdateTextureKernel( rt::RenderData& renderData )
{
  dim3 threadsPerBlock( 32, 32, 1 );
  dim3 blocksPerGrid( renderData.mDimensions.x / threadsPerBlock.x
                      , renderData.mDimensions.y / threadsPerBlock.y, 1 ); // TODO works only with power of 2 texture sizes !!
  if ( blocksPerGrid.x == 0 )
  {
    blocksPerGrid.x = 1;
  }
  if ( blocksPerGrid.y == 0 )
  {
    blocksPerGrid.y = 1;
  }

  cudaEvent_t start, stop;
  cudaEventCreate( &start );
  cudaEventCreate( &stop );

  cudaEventRecord( start, 0 );
  UpdateTextureKernel<<<blocksPerGrid, threadsPerBlock>>> ( renderData.mPixelBuffer, renderData.mDimensions.x, renderData.mDimensions.y );
  cudaEventRecord( stop, 0 );
  cudaEventSynchronize( stop );

  float time = 0.0f;
  cudaEventElapsedTime( &time, start, stop );
  return cudaGetLastError();
}
