#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand.h>
#include <curand_kernel.h>

#include "Common/Math.h"

namespace utils
{
inline __device__ uint8_t Component( const uint32_t& color, const uint32_t& idx )
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

inline __device__ uint32_t Color( const uint8_t r = 0, const uint8_t g = 0, const uint8_t b = 0, const uint8_t a = 255 )
{
  return ( r << 0 ) | ( g << 8 ) | ( b << 16 ) | ( a << 24 );
}

/// Allocate on the GPU
// cudaMalloc((void**) &RandomStates, width * height * sizeof(curandState_t));
/// Initialize
// InitRandom<<<gridSize, blockSize>>>( static_cast<uint32_t>( time( nullptr ) ), width, height, RandomStates );

/// initialize each states in RandomStates:: mStates by calling this kernel on them:
// __global__ void InitRandom( unsigned int seed, const uint32_t width, const uint32_t height, curandState_t* states )
// {
//   const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
//   const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
//   if ( x >= width || y >= height )
//   {
//     return;
//   }
// 
//   /* we have to initialize the state */
//   curand_init( seed, /* the seed can be the same for each core, here we pass the time in from the CPU */
//                x, /* the sequence number should be different for each core (unless you want all
//                   cores to get the same sequence of numbers for some reason - use thread id! */
//                y, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
//                &states[x + y * width] );
// }

/// User Kernel needs the states
// .. , curandState_t* states, ..
// RenderKernel<<<>>>( .. , RandomStates.mStates, ..

/// use them in curand_uniform calls
// const float r = curand_uniform( &states[x + y * width] );

// TODO can we reuse the same state between frames or we need new states for each frame?
//class RandomStates
//{
//public:
//  __device__ RandomStates( const uint32_t width, const uint32_t height )
//  {
//    cudaMalloc( (void**) &mStates, width * height * sizeof( curandState_t ) );
//    // InitRandom<<<gridSize, blockSize>>>( static_cast<uint32_t>( time( nullptr ) ), width, height, gRandomStates );
//  }
//
//  __device__ ~RandomStates()
//  {
//    cudaFree( mStates );
//  }
//
//  curandState_t* mStates = nullptr;
//};

inline __device__ __host__ float Random( float min = 0.0, float max = 1.0 )
{
  return min; // TODO: just to compile
}

inline __device__ __host__ math::vec2 RandomOnCircle()
{
  const auto t( 2.0f * 3.14156545f * Random() );
  const auto u( Random() + Random() );
  const auto sr( u > 1.0 ? 2.0 - u : u );
  return math::vec2( sr * glm::cos( t ), sr * glm::sin( t ) );
}

} // ::utils
