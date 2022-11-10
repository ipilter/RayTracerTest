#include "Camera.h"

namespace gl
{

Camera::Camera( const math::vec3& center )
  : mProjectionMatrix( 1.0 )
  , mViewMatrix( 1.0 )
{
  Center( center );
}

Camera::~Camera()
{ }

math::mat4 Camera::ViewProj() const
{
  return mProjectionMatrix * mViewMatrix;
}

void Camera::Ortho( const float xSpan, const float ySpan )
{
  mProjectionMatrix = glm::ortho( -1.0f * xSpan, xSpan, -1.0f * ySpan, ySpan );
}

void Camera::Translate( const math::vec3& v )
{
  mViewMatrix = glm::translate( mViewMatrix, v );
}

void Camera::Scale( const math::vec3& v )
{
  mViewMatrix = glm::scale( mViewMatrix, math::vec3( v.x, v.y, v.z ) );
}

void Camera::Center( const math::vec3& v )
{
  mViewMatrix = glm::translate( glm::identity<math::mat4>(), v );
}

}
