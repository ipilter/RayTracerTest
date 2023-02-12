#pragma once
#include "Common\Math.h"
#include <cuda_runtime.h>

namespace rt
{
// TODO solve this copy-paste mess 
inline void CreateRenderBuffer( const math::uvec2& bufferSize
                                , const uint32_t channelCount
                                , float*& renderBuffer )
{
  const size_t byteCount = bufferSize.x * bufferSize.y * channelCount * sizeof( float );
  cudaError_t err = cudaMalloc( reinterpret_cast<void**>( &renderBuffer ), byteCount );
  if ( err != cudaSuccess )
  {
    throw std::runtime_error( std::string( __FUNCTION__ "cudaMalloc failed. (" ) + cudaGetErrorString( err ) + ")\n");
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
    throw std::runtime_error( std::string( __FUNCTION__ "cudaMemset failed. (" ) + cudaGetErrorString( err ) + ")\n");
  }
}

inline void CreateSampleCountBuffer( const math::uvec2& bufferSize
                                     , uint32_t*& sampleCountBuffer )
{
  const size_t byteCount = bufferSize.x * bufferSize.y * sizeof( uint32_t );
  cudaError_t err = cudaMalloc( reinterpret_cast<void**>( &sampleCountBuffer ), byteCount );
  if ( err != cudaSuccess )
  {
    throw std::runtime_error( std::string( __FUNCTION__ "cudaMalloc failed. (" ) + cudaGetErrorString( err ) + ")\n");
  }
}

inline void ClearSampleCountBuffer( const math::uvec2& bufferSize
                                    , uint32_t*& sampleCountBuffer )
{
  const size_t byteCount = bufferSize.x * bufferSize.y * sizeof( uint32_t );
  cudaError_t err = cudaMemset( sampleCountBuffer, 0, byteCount );
  if ( err != cudaSuccess )
  {
    throw std::runtime_error( std::string( __FUNCTION__ "cudaMemset failed. (" ) + cudaGetErrorString( err ) + ")\n");
  }
}

inline void CreateImageBuffer( const math::uvec2& bufferSize
                               , rt::Color*& imageBuffer )
{
  const size_t byteCount = bufferSize.x * bufferSize.y * sizeof( rt::Color );
  cudaError_t err = cudaMalloc( reinterpret_cast<void**>( &imageBuffer ), byteCount );
  if ( err != cudaSuccess )
  {
    throw std::runtime_error( std::string( __FUNCTION__ "cudaMalloc failed. (" ) + cudaGetErrorString( err ) + ")\n");
  }
}

inline void ClearImageBuffer( const math::uvec2& bufferSize
                              , rt::Color*& imageBuffer )
{
  const size_t byteCount = bufferSize.x * bufferSize.y * sizeof( rt::Color );
  cudaError_t err = cudaMemset( imageBuffer, 0, byteCount );
  if ( err != cudaSuccess )
  {
    throw std::runtime_error( std::string( __FUNCTION__ "cudaMemset failed. (" ) + cudaGetErrorString( err ) + ")\n");
  }
}

// devicePtr : GPU memory pointer
// hostPtr : CPU memory pointer
// count: element count in the buffer
template<class T>
inline void CopyDeviceDataToHost( const T* devicePtr, T* hostPtr, const size_t count )
{
  cudaError_t err = cudaMemcpy( hostPtr, devicePtr, count * sizeof( T ), cudaMemcpyDeviceToHost );
  if ( err != cudaSuccess )
  {
    throw std::runtime_error( std::string( __FUNCTION__ "cudaMemcpy failed. (" ) + cudaGetErrorString( err ) + ")\n");
  }
}

}
