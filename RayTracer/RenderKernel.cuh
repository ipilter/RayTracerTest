#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>

#include "Common\Math.h"

#include "ThinLensCamera.cuh"

namespace rt
{

// Note: arguments MUST be by value or by pointer. Pointer MUST be in device mem space
__global__ void RenderKernel( uint32_t* pixelBufferPtr
                                     , const math::uvec2 pixelBufferSize
                                     , rt::ThinLensCamera camera
                                     , const uint32_t sampleCount
                                     , curandState_t* randomStates )
{
  using namespace math;

  const uvec2 pixel( blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y );
  if ( pixel.x >= pixelBufferSize.x || pixel.y >= pixelBufferSize.y )
  {
    return;
  }

  const uint32_t offset( pixel.x + pixel.y * pixelBufferSize.x );
  curandState_t randomState = randomStates[offset];

  vec3 accu( 0.0f );
  for ( auto s( 0 ); s < sampleCount; ++s )
  {
    const rt::Ray ray( camera.GetRay( pixel, pixelBufferSize, randomState ) );
    accu += ray.direction();
  }
  accu /= sampleCount;

  // save final pixel color
  pixelBufferPtr[offset] = utils::Color( 255 * accu.x
                                         , 255 * accu.y
                                         , 255 * accu.z );

  randomStates[offset] = randomState;
}

}
