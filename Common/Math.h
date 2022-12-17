#pragma once

#include <iomanip>
#include <random>

// TODO: glm in CUDA emits tonns of warngings about annotated defaulted/deleted methds.. 
//       warnings are supressed in CUDA props
#define GLM_FORCE_XYZW_ONLY
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/quaternion.hpp>

namespace math
{

using ivec2 = glm::ivec2;
using uvec2 = glm::uvec2;
using uvec3 = glm::uvec3;

using vec2 = glm::vec2;
using vec3 = glm::vec3;
using vec4 = glm::vec4;

using mat4 = glm::mat4;

using quat = glm::quat;

constexpr float cPi = 3.141592653589793f;
constexpr float cEps = 1e-9f;

// TODO T can be float or double only!
template<class T>
inline T Random( T min = 0, T max = 1 )
{
  static std::random_device rd;
  static std::mt19937 gen( rd() );
  static std::uniform_real_distribution<T> dist( min, max );

  return dist( gen );
}

inline quat MakeQuaternion( const vec3& axis, const float angle )
{
  const float halfAngle = angle / 2.0f;
  const float sinAngle = glm::sin( halfAngle );
  const vec3 realPart( axis * sinAngle );
  return quat( glm::cos( halfAngle ), realPart );
}

inline math::vec3 Rotate( const vec3& vector, const vec3& axis, const float angle )
{
  const glm::quat q( glm::normalize( MakeQuaternion( axis, angle ) ) );
  return q * vector;
}

}

inline std::ostream& operator << ( std::ostream& stream, const math::ivec2& v )
{
  stream << "[" << v.x << ", " << v.y << "]";
  return stream;
}

inline std::ostream& operator << ( std::ostream& stream, const math::uvec2& v )
{
  stream << "[" << v.x << ", " << v.y << "]";
  return stream;
}

inline std::ostream& operator << ( std::ostream& stream, const math::vec2& v )
{
  stream << std::fixed << std::setprecision( 6 ) << "[" << v.x << ", " << v.y << "]";
  return stream;
}

inline std::ostream& operator << ( std::ostream& stream, const math::vec3& v )
{
  stream << std::fixed << std::setprecision( 6 ) << "[" << v.x << ", " << v.y << ", " << v.z << "]";
  return stream;
}

inline std::ostream& operator << ( std::ostream& stream, const math::vec4& v )
{
  stream << std::fixed << std::setprecision( 6 ) << "[" << v.x << ", " << v.y << ", " << v.z << ", " << v.w << "]";
  return stream;
}

inline std::ostream& operator << ( std::ostream& stream, const math::mat4& m )
{
  stream << std::fixed << std::setprecision( 6 ) << "[";
  for ( auto i = 0; i < 4; ++i )
  {
    if ( i != 0 )
    {
      stream << ", ";
    }
    stream << "[";
    for ( auto j = 0; j < 4; ++j )
    {
      if ( j != 0 )
      {
        stream << ", ";
      }
      stream << m[i][j];
    }
    stream << "]";
  }
  stream << "]";
  return stream;
}
