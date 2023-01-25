#pragma once

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "Common\Math.h"

namespace random
{

__host__ void CreateStates( const math::uvec2& size, curandState_t*& states );

inline __device__ math::vec2 UnifromOnDisk( curandState_t& randomState )
{
  const float t( 2.0f * 3.14156545f * curand_uniform( &randomState ) );
  const float u( curand_uniform( &randomState ) + curand_uniform( &randomState ) );
  const float sr( u > 1.0f ? 2.0f - u : u );
  return math::vec2( sr * glm::cos( t ), sr * glm::sin( t ) );
}

inline __device__ float Uniform( curandState_t& randomState )
{
  return curand_uniform( &randomState );
}

}
