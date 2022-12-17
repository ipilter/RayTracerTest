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
  RayTracer( const math::uvec2& pixelBufferSize
             , const float fov
             , const float focalLength
             , const float aperture );
  ~RayTracer();

  void Trace( rt::color_t* pixelBufferPtr, const uint32_t sampleCount );
  void Resize( const math::uvec2& size );
  void SetCameraParameters( const float fov
                            , const float focalLength
                            , const float aperture );
  void RotateCamera( const math::vec2& angles );

private:
  RayTracerImpl* mImpl;
};

}
