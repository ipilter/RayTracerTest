#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "Common\Math.h"

namespace rt
{
class Ray
{
public:
  __host__ __device__ Ray( const math::vec3& o = math::vec3( 0.0f )
                           , const math::vec3& d = math::vec3( 0.0f )
                           , const bool normalizeDirection = true )
    : mOrigin( o )
    , mDirection( normalizeDirection ? glm::normalize( d ) : d )
  {}

  __host__ __device__ Ray( const Ray& rhs )
    : mOrigin( rhs.mOrigin )
    , mDirection( rhs.mDirection )
  {}

  __host__ __device__ Ray& operator = ( const Ray& rhs )
  {
    mOrigin = rhs.mOrigin;
    mDirection = rhs.mDirection;
    return *this;
  }

  __host__ __device__ const math::vec3& origin() const
  {
    return mOrigin;
  }
  
  __host__ __device__ const math::vec3& direction() const
  {
    return mDirection;
  }

  __host__ __device__ math::vec3 point( const float t ) const
  {
    return mOrigin + mDirection * t;
  }

private:
  math::vec3 mOrigin;
  math::vec3 mDirection;
};

}
