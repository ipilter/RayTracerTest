#pragma once

#include <iomanip>

// TODO: glm in CUDA emits tonns of warngings about annotated defaulted/deleted methds.. 
//       warning supressed in CUDA props
#define GLM_FORCE_XYZW_ONLY
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace math
{

using ivec2 = glm::ivec2;
using uvec2 = glm::uvec2;
using uvec3 = glm::uvec3;

using vec2 = glm::vec2;
using vec3 = glm::vec3;
using vec4 = glm::vec4;

using mat4 = glm::mat4;

constexpr float cPi = 3.141592653589793f;
constexpr float cEps = 1e-9f;

float rad( const float deg );

}

std::ostream& operator << ( std::ostream& stream, const math::ivec2& v );
std::ostream& operator << ( std::ostream& stream, const math::uvec2& v );
std::ostream& operator << ( std::ostream& stream, const math::vec2& v );
std::ostream& operator << ( std::ostream& stream, const math::vec3& v );
std::ostream& operator << ( std::ostream& stream, const math::vec4& v );
std::ostream& operator << ( std::ostream& stream, const math::mat4& m );
