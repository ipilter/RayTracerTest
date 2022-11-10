#include "Math.h"

namespace math
{

float rad( const float deg )
{
  constexpr float factor = cPi / 180.0f;
  return deg * factor;
}

}

std::ostream& operator << ( std::ostream& stream, const math::ivec2& v )
{
  stream << "[" << v.x << ", " << v.y << "]";
  return stream;
}

std::ostream& operator << ( std::ostream& stream, const math::uvec2& v )
{
  stream << "[" << v.x << ", " << v.y << "]";
  return stream;
}

std::ostream& operator << ( std::ostream& stream, const math::vec2& v )
{
  stream << std::fixed << std::setprecision(6) << "[" << v.x << ", " << v.y << "]";
  return stream;
}

std::ostream& operator << ( std::ostream& stream, const math::vec3& v )
{
  stream << std::fixed << std::setprecision(6) << "[" << v.x << ", " << v.y << ", " << v.z << "]";
  return stream;
}

std::ostream& operator << ( std::ostream& stream, const math::vec4& v )
{
  stream << std::fixed << std::setprecision(6) << "[" << v.x << ", " << v.y << ", " << v.z << ", " << v.w << "]";
  return stream;
}

std::ostream& operator << ( std::ostream& stream, const math::mat4& m )
{
  stream << std::fixed << std::setprecision(6) << "[";
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

