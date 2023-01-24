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

  const uint32_t offset( pixel.x + pixel.y * bufferSize.x );
  curandState_t randomState = randomStates[offset];

  const uint32_t offset2( channelCount * pixel.x + pixel.y * bufferSize.x * channelCount );

  // TODO do not calculate the average here, render data must be a more-than-float-pixel-value array 
  // and store the corrent accumulated value and the total samples used separately
  // in the render -> image data conversion will do the average counting
  vec3 accu( 0.0f );
  for ( auto s( 0 ); s < sampleCount; ++s )
  {
    const rt::Ray ray( camera.GetRay( pixel, bufferSize, randomState ) );
    accu += ray.direction();
  }
  accu /= static_cast<float>( sampleCount );

  // save final pixel color
  renderBuffer[offset2] = ( renderBuffer[offset2] + accu.x ) / 2.0f;
  renderBuffer[offset2 + 1] = ( renderBuffer[offset2 + 1] + accu.y ) / 2.0f;
  renderBuffer[offset2 + 2] = ( renderBuffer[offset2 + 2] + accu.z ) / 2.0f;
  //renderBuffer[offset2+3] += ?;

  randomStates[offset] = randomState;
}

__global__ void ConverterKernel( const math::uvec2 bufferSize
                                 , const uint32_t channelCount
                                 , float* renderBuffer
                                 , rt::Color* imageBuffer )
{
  using namespace math;

  const uvec2 pixel( blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y );
  if ( pixel.x >= bufferSize.x || pixel.y >= bufferSize.y )
  {
    return;
  }

  const uint32_t offset( pixel.x + pixel.y * bufferSize.x );
  const uint32_t offset2( channelCount * pixel.x + pixel.y * bufferSize.x * channelCount );
  imageBuffer[offset] = utils::GetColor( 255u * renderBuffer[offset2]
                                      , 255u * renderBuffer[offset2 + 1]
                                      , 255u * renderBuffer[offset2 + 2] );
}

}
