#pragma once

#include "..\Math.h"

namespace rt
{
class Ray
{
public:
  Ray( const math::vec3& o
       , const math::vec3& d
       , const bool normalizeDirection = true )
    : mOrigin( o )
    , mDirection( normalizeDirection ? glm::normalize( d ) : d )
  {}

  const math::vec3& origin() const
  {
    return mOrigin;
  }
  const math::vec3& direction() const
  {
    return mDirection;
  }

  math::vec3 point( const float t ) const
  {
    return mOrigin + mDirection * t;
  }

private:
  math::vec3 mOrigin;
  math::vec3 mDirection;
};

}
