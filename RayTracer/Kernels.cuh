#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>

#include "Common\Math.h"
#include "Common\Color.h"

#include "ThinLensCamera.cuh"

namespace rt
{

// Note: arguments MUST be by value or by pointer. Pointer MUST be in device mem space
__global__ void TraceKernel( float* renderBuffer
                             , uint32_t* sampleCountBuffer
                             , const math::uvec2 bufferSize
                             , const uint32_t channelCount
                             , rt::ThinLensCamera camera
                             , const uint32_t sampleCount
                             , curandState_t* randomStates )
{
  using namespace math;

  const uvec2 pixel( blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y );
  if ( pixel.x >= bufferSize.x || pixel.y >= bufferSize.y )
  {
    return;
  }

  const uint32_t pixelOffset( pixel.x + pixel.y * bufferSize.x );
  const uint32_t valueOffset( channelCount * pixel.x + pixel.y * bufferSize.x * channelCount );

  curandState_t randomState = randomStates[pixelOffset];

  math::vec3 accu(0.0f);
  for ( auto s( 0 ); s < sampleCount; ++s )
  {
    const rt::Ray ray( camera.GetRay( pixel, bufferSize, randomState ) );
    accu += ray.direction();
  }

  sampleCountBuffer[pixelOffset] += sampleCount;
  renderBuffer[valueOffset + 0] += accu.x;
  renderBuffer[valueOffset + 1] += accu.y;
  renderBuffer[valueOffset + 2] += accu.z;
  //renderBuffer[valueOffset + 3] = ;

  randomStates[pixelOffset] = randomState;
}

__global__ void ConverterKernel( const math::uvec2 bufferSize
                                 , const uint32_t channelCount
                                 , float* renderBuffer
                                 , uint32_t* sampleCountBuffer
                                 , rt::Color* imageBuffer )
{
  using namespace math;

  const uvec2 pixel( blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y );
  if ( pixel.x >= bufferSize.x || pixel.y >= bufferSize.y )
  {
    return;
  }

  const uint32_t pixelOffset( pixel.x + pixel.y * bufferSize.x );
  const uint32_t valueOffset( channelCount * pixel.x + pixel.y * bufferSize.x * channelCount );
  const float sampleCount( static_cast<float>( sampleCountBuffer[pixelOffset] ) );
  imageBuffer[pixelOffset] = utils::GetColor(   255u * ( renderBuffer[valueOffset + 0] / sampleCount )
                                              , 255u * ( renderBuffer[valueOffset + 1] / sampleCount )
                                              , 255u * ( renderBuffer[valueOffset + 2] / sampleCount ) );
}

}
