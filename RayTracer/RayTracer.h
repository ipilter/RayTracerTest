#pragma once

#include "..\Sptr.h"
#include "..\OpenGL\PixelBufferObject.h"

namespace rt
{

class RayTracer : public ISptr<RayTracer>
{
public:
  RayTracer();
  void Trace( gl::PixelBufferObject::sptr& pbo, const math::uvec2& pixelsSize );

private:
};

}
