#include "Random.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace rt
{

__global__ void InitRandom( uint32_t seed
                            , const uint32_t width
                            , const uint32_t height
                            , curandState_t* states )
{
  const uint32_t x( blockIdx.x * blockDim.x + threadIdx.x );
  const uint32_t y( blockIdx.y * blockDim.y + threadIdx.y );
  if ( x >= width || y >= height )
  {
    return;
  }

  const size_t offset( x + y * width );
  curandState_t state( states[offset] );

  curand_init( seed,   // the seed can be the same for each core
               offset, // the sequence number should be different for each core (unless you want all cores to get the same sequence of numbers for some reason - use thread id!
               0,      // the offset is how much extra we advance in the sequence for each call, can be 0
               &state );

  states[offset] = state;
}

void Random::Init( const math::uvec2& size )
{
  cudaError_t err = cudaMalloc( reinterpret_cast<void**>( &mStates ), size.x * size.y * sizeof( curandState_t ) );
  if ( err != cudaSuccess )
  {
    // TODO handle error
    const auto str = cudaGetErrorString( err );
  }

  const dim3 threadsPerBlock( 32, 32, 1 );
  const dim3 blocksPerGrid( static_cast<uint32_t>( glm::ceil( size.x / static_cast<float>( threadsPerBlock.x ) ) )
                            , static_cast<uint32_t>( glm::ceil( size.y / static_cast<float>( threadsPerBlock.y ) ) )
                            , 1 );

  InitRandom<<<blocksPerGrid, threadsPerBlock>>>( static_cast<uint32_t>( time( nullptr ) ), size.x, size.y, mStates );
  err = cudaGetLastError();
  if ( err != cudaSuccess )
  {
    // TODO handle error
    const auto str = cudaGetErrorString( err );
  }
}

}
