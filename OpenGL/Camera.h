#pragma once

#include "..\Math.h"
#include "..\Sptr.h"

namespace gl
{

class Camera : public virtual ISptr<Camera>
{
public:
  Camera( const math::vec3& center );
  ~Camera();

  math::mat4 ViewProj() const;
  void Ortho( const float xSpan, const float ySpan );
  void Translate( const math::vec3& v );
  void Scale( const math::vec3& v );
  void Center( const math::vec3& v );

private:
  math::mat4 mProjectionMatrix;
  math::mat4 mViewMatrix;
};

}
