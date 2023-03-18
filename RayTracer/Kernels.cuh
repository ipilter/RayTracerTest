#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include <texture_fetch_functions.h>

#include "Common\Math.h"
#include "Common\Color.h"

#include "ThinLensCamera.cuh"

namespace rt
{

__device__ float abs( float f )
{
  return (f < 0.0f) ? -f : f;
}

__device__ math::vec3& abs( math::vec3& v )
{
  v.x = abs( v.x );
  v.y = abs( v.y );
  v.z = abs( v.z );
  return v;
}

__device__ bool HitTriangle( const Ray& ray
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

// see https://developer.nvidia.com/blog/accelerated-ray-tracing-cuda/
__device__ math::vec3 Radiance( const Ray& ray, const math::vec3& background, cudaTextureObject_t sceneTrianglesTextureObject, uint32_t numberOfTriangles )
{
  math::vec3 ret( background );

  // search for the closest triangle we hit with this ray
  float distance = -FLT_MAX;
  math::vec3 ta, tb, tc;
  for ( uint32_t triangleIdx = 0; triangleIdx < numberOfTriangles; ++triangleIdx )
  {
    const uint32_t triangleOffset = 3 * triangleIdx;
    const float4 a4 = tex2D<float4>( sceneTrianglesTextureObject, triangleOffset + 0.0f, 0.0f );
    const float4 b4 = tex2D<float4>( sceneTrianglesTextureObject, triangleOffset + 1.0f, 0.0f );
    const float4 c4 = tex2D<float4>( sceneTrianglesTextureObject, triangleOffset + 2.0f, 0.0f );
    const math::vec3 a( a4.x, a4.y, a4.z ), b( b4.x, b4.y, b4.z ), c( c4.x, c4.y, c4.z );

    float t( 0.0f ), u( 0.0f ), v( 0.0f );
    if ( HitTriangle( ray, a, b, c, t, u, v ) && distance < t )
    {
      distance = t;

      ta = a;
      tb = b;
      tc = c;
    }
  }

  // if we hit a triangle, calculate it's contribution for the final color, return background otherwise
  if ( distance > -FLT_MAX )
  {
    const math::vec3 n( glm::normalize( glm::cross( tb - ta, tc - ta ) ) );
    const math::vec3 hitpoint( ray.point( distance ) );
    ret = glm::abs( n );
  }
  else
  {
    ret = background * 0.8f + ray.direction() * 0.2f;
  }

  return ret;
}

// Note: arguments MUST be by value or by pointer. Pointer MUST be in device mem space
__global__ void TraceKernel( float* renderBuffer
                             , uint32_t* sampleCountBuffer
                             , const math::uvec2 bufferSize
                             , const uint32_t channelCount
                             , rt::ThinLensCamera camera
                             , const uint32_t sampleCount
                             , curandState_t* randomStates
                             , cudaTextureObject_t sceneTrianglesTextureObject
                             , uint32_t numberOfTriangles )
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
  const math::vec3 background( 0.15f, 0.11f, 0.13f );
  math::vec3 accu( 0.0f );
  for ( auto s( 0 ); s < sampleCount; ++s )
  {
    const rt::Ray ray( camera.GetRay( pixel, bufferSize, randomState ) );
    accu += Radiance( ray, background, sceneTrianglesTextureObject, numberOfTriangles);
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
