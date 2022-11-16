#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand.h>
#include <curand_kernel.h>

#include "Common\Math.h"

namespace rt
{
__global__ void InitRandom( uint32_t seed, uint32_t width, uint32_t height, curandState_t* states );

class Random
{
public:
  Random( const math::uvec2& size )
    : mStates( nullptr )
  {
    Init( size );
  }

  ~Random()
  {
    const cudaError_t err = cudaFree( mStates );
    if ( err != cudaSuccess )
    {
      // TODO handle error
      const auto str = cudaGetErrorString( err );
    }
  }

  // TODO do it safer, better
  __device__ void SetOffset( const uint32_t o )
  {
    mOffset = o;
  }

  // TODO unresolved external symbo if these are in the cu file!
  __device__ float Uniform()
  {
    return curand_uniform( &mStates[mOffset] );
  }

  __device__ math::vec2 UnifromOnDisk()
  {
    //const float t( 2.0f * 3.14156545f * curand_uniform( &mStates[mOffset] ) );
    //const float u( curand_uniform( &mStates[mOffset] ) + curand_uniform( &mStates[mOffset] ) );
    //const float sr( u > 1.0f ? 2.0f - u : u );
    //return math::vec2( sr * glm::cos( t ), sr * glm::sin( t ) );
    return math::vec2( curand_uniform( &mStates[mOffset] ), curand_uniform( &mStates[mOffset] ) );
  }

private:
  void Init( const math::uvec2& size );

  curandState_t* mStates;
  uint32_t mOffset;
};

}
