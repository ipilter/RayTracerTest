#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>

#include "Common\Math.h"
#include "Common\Color.h"

#include "ThinLensCamera.cuh"
#include <thrust/device_vector.h>

namespace rt
{

__host__ __device__ bool Hit( const Ray& ray
                              , const math::vec3& v0
                              , const math::vec3& v1
                              , const math::vec3& v2
                              , float& t
                              , float& u
                              , float& v )
{
  const math::vec3 v0v1( v1 - v0 );
  const math::vec3 v0v2( v2 - v0 );
  const math::vec3 pv( glm::cross( ray.direction(), v0v2 ) );
  const float det( glm::dot( v0v1, pv ) );

  if ( det < 0.0000000001f )
  {
    return false; // backfacing triangle not visible (Culling)
  }

  const float invDet( 1.0f / det );

  const math::vec3 tv( ray.origin() - v0 );
  u = glm::dot( tv, pv ) * invDet;
  if ( u < 0.0f || u > 1.0f )
  {
    return false;
  }

  const math::vec3 qv( glm::cross( tv, v0v1 ) );
  v = glm::dot( ray.direction(), qv ) * invDet;
  if ( v < 0.0f || u + v > 1.0f )
  {
    return false;
  }

  t = glm::dot( v0v2, qv ) * invDet;
  return true;
}

// Triangle with a b c vertices
// create only on device and add ability to get data from host
// initialize with a big chunk of memory, follow aaa|bbb|ccc order instead of abcabcabc
class Triangle
{
public:
  __host__ __device__ Triangle( const uint64_t* data )
    : mDataPtr( data )
  { }

  __host__ __device__ const uint64_t& a() const
  {
    return *mDataPtr;
  }

  __host__ __device__ const uint64_t& b() const
  {
    return *( mDataPtr + 1 );
  }

  __host__ __device__ const uint64_t& c() const
  {
    return *( mDataPtr + 2 );
  }

private:
  const uint64_t* mDataPtr; // TODO use mData, mData+1, mData+2 as a,b,c but nicer than this
};

// create only on device and add ability to get data from host
// initialize with a big chunk of memory, follow xxx|yyy|zzz order instead of xyzxyzxyz
//
// data layout:
// [x0,y0,z0][x1,y1,z1]..[xN,yN,zN]
class VertexPool
{
public:
  __host__ __device__ VertexPool( const float* data )
    : mDataPtr( data )
  { }

  __host__ __device__ bool triangle( const Triangle& triangle, math::vec3& a, math::vec3& b, math::vec3& c ) const
  {
    a = math::vec3( x( triangle.a() ), y( triangle.a() ), z( triangle.a() ) );
    b = math::vec3( x( triangle.b() ), y( triangle.b() ), z( triangle.b() ) );
    c = math::vec3( x( triangle.c() ), y( triangle.c() ), z( triangle.c() ) );
    return true;
  }

private:
  __host__ __device__ const float& x( const uint64_t& idx ) const 
  {
    return *( mDataPtr + idx * 3 );
  }

  __host__ __device__ const float& y( const uint64_t& idx ) const
  {
    return *( mDataPtr + (idx * 3) + 1);
  }

  __host__ __device__ const float& z( const uint64_t& idx ) const
  {
    return *( mDataPtr + (idx * 3) + 2);
  }

private:
  const float* mDataPtr;
};

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

  curandState_t randomState = randomStates[pixelOffset]; // for faster performance we make a copy in the fast memory, save it later


  math::vec3 background( 0.1f, 0.1f, 0.15f );
  
  const float vertices[] = {0.0f, 0.0f, 10.0f,  1.0f, 0.0f, 10.0f,  0.0f, 1.0f, 10.0f};

  VertexPool vertexPool( vertices );
  const uint64_t triangleData[] = { 0, 2, 1 };
  Triangle trianglePool[] = { Triangle( triangleData ) };

  math::vec3 accu( 0.0f );
  for ( auto s( 0 ); s < sampleCount; ++s )
  {
    const rt::Ray ray( camera.GetRay( pixel, bufferSize, randomState ) );

    math::vec3 a, b, c;
    vertexPool.triangle( trianglePool[0], a, b, c );

    float t( 0.0f ), u( 0.0f ), v( 0.0f );
    const bool hit = Hit( ray, a, b, c, t, u, v );
    if ( hit )
    {
      const math::vec3 n( glm::normalize( glm::cross( b - a, c - a ) ) );
      const math::vec3 hitpoint( ray.point( t ) );

      const math::vec3 color = glm::abs( n );

      accu += color;
    }
    else
    {
      accu += ray.direction() * 0.2f;
      //accu += background;
    }
  }

  sampleCountBuffer[pixelOffset] += sampleCount;
  renderBuffer[valueOffset + 0] += accu.x;
  renderBuffer[valueOffset + 1] += accu.y;
  renderBuffer[valueOffset + 2] += accu.z;
  //renderBuffer[valueOffset + 3] += accu.w;

  randomStates[pixelOffset] = randomState; // save random state back into main memory
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
