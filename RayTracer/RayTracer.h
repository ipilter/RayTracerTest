#pragma once

#include "Common\Math.h"
#include "Common\Sptr.h"
#include "Common\Color.h"

namespace rt
{
class RayTracerImpl;

class RayTracer : public ISptr<RayTracer>
{
public:
  RayTracer( const math::uvec2& pixelBufferSize );
  ~RayTracer();

  void Trace( rt::color_t* pixelBufferPtr
              , const uint32_t sampleCount
              , const float fov
              , const float focalLength
              , const float aperture );

  void Resize( const math::uvec2& size );

private:
  RayTracerImpl* mImpl;
};

}
