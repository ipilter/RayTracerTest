#pragma once
#include "Common\Math.h"
#include <cuda_runtime.h>

namespace render
{

inline void CreateRenderBuffer( const math::uvec2& bufferSize
                                , const uint32_t channelCount
                                , float*& renderBuffer )
{
  const size_t byteCount = bufferSize.x * bufferSize.y * channelCount * sizeof( float );
  cudaError_t err = cudaMalloc( reinterpret_cast<void**>( &renderBuffer ), byteCount );
  if ( err != cudaSuccess )
  {
    throw std::runtime_error( std::string( "cudaMalloc failed. (" ) + cudaGetErrorString( err ) + ")\n");
  }
}

inline void ClearRenderBuffer( const math::uvec2& bufferSize
                               , const uint32_t channelCount
                               , float*& renderBuffer )
{
  const size_t byteCount = bufferSize.x * bufferSize.y * channelCount * sizeof( float );
  cudaError_t err = cudaMemset( renderBuffer, 0, byteCount );
  if ( err != cudaSuccess )
  {
    throw std::runtime_error( std::string( "cudaMemset failed. (" ) + cudaGetErrorString( err ) + ")\n");
  }
}

inline void CreateSampleCountBuffer( const math::uvec2& bufferSize
                                     , uint32_t*& sampleCountBuffer )
{
  const size_t byteCount = bufferSize.x * bufferSize.y * sizeof( uint32_t );
  cudaError_t err = cudaMalloc( reinterpret_cast<void**>( &sampleCountBuffer ), byteCount );
  if ( err != cudaSuccess )
  {
    throw std::runtime_error( std::string( "cudaMalloc failed. (" ) + cudaGetErrorString( err ) + ")\n");
  }
}

inline void ClearSampleCountBuffer( const math::uvec2& bufferSize
                                    , uint32_t*& sampleCountBuffer )
{
  const size_t byteCount = bufferSize.x * bufferSize.y * sizeof( uint32_t );
  cudaError_t err = cudaMemset( sampleCountBuffer, 0, byteCount );
  if ( err != cudaSuccess )
  {
    throw std::runtime_error( std::string( "cudaMemset failed. (" ) + cudaGetErrorString( err ) + ")\n");
  }
}

inline void CreateImageBuffer( const math::uvec2& bufferSize
                               , rt::Color*& imageBuffer )
{
  const size_t byteCount = bufferSize.x * bufferSize.y * sizeof( rt::Color );
  cudaError_t err = cudaMalloc( reinterpret_cast<void**>( &imageBuffer ), byteCount );
  if ( err != cudaSuccess )
  {
    throw std::runtime_error( std::string( "cudaMalloc failed. (" ) + cudaGetErrorString( err ) + ")\n");
  }
}

inline void ClearImageBuffer( const math::uvec2& bufferSize
                              , rt::Color*& imageBuffer )
{
  const size_t byteCount = bufferSize.x * bufferSize.y * sizeof( rt::Color );
  cudaError_t err = cudaMemset( imageBuffer, 0, byteCount );
  if ( err != cudaSuccess )
  {
    throw std::runtime_error( std::string( "cudaMemset failed. (" ) + cudaGetErrorString( err ) + ")\n");
  }
}

}
