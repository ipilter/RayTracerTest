#include "pch.h"

#include <Common\Math.h>


// Lazy copy-past code here
// TODO do not copy paste the code but let the unit test access the hidden part of the RayTracer project and use the cu/cuh files directly.
namespace rt
{
float pack( const math::vec3& normal )
{
	return glm::floor( normal.x * 127.0f + 127.5f ) / 256.0f +		// 2^8
				 glm::floor( normal.y * 127.0f + 127.5f ) / 65'536.0f +		// 2^16
				 glm::floor( normal.z * 127.0f + 127.5f ) / 16'777'216.0f;	// 2^24
}

math::vec3 unpack(const float packedNormal)
{
	const float OneOver127 = 1.0f / 127.0f;
	return math::vec3(glm::floor(glm::fract(packedNormal * 1.0f) * 256.0f) * OneOver127 - 1.0f,
									  glm::floor(glm::fract(packedNormal * 65'536.0f) * 256.0f) * OneOver127 - 1.0f,
									  glm::floor(glm::fract(packedNormal * 16'777'216.0f) * 256.0f) * OneOver127 - 1.0f);
}

}

// ray from 0,0,0 towards 0,0,-1 hits the triangle at z=-10 with hit point 0,0,-10
TEST( TestCaseName, pack_unpack )
{
	const math::vec3 normal( 0.0f, 0.0f, 1.0f );
	const float packed( rt::pack( normal ) );
	const math::vec3 unpacked( rt::unpack( packed ) );
	EXPECT_EQ( normal, unpacked );
}
