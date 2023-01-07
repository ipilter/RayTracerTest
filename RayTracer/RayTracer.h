#pragma once

#include <functional>
#include <driver_types.h>

#include "Common\Math.h"
#include "Common\Sptr.h"
#include "Common\Color.h"

namespace rt
{
class RayTracerImpl;

class RayTracer : public ISptr<RayTracer>
{
public:
  using CallBackFunction = std::function<void()>;

public:
  RayTracer( const math::uvec2& pixelBufferSize
             , const math::vec3& cameraPosition
             , const math::vec2& cameraAngles
             , const float fov
             , const float focalLength
             , const float aperture );
  ~RayTracer();

  void Trace( cudaGraphicsResource_t pboCudaResource
              , const uint32_t iterationCount
              , const uint32_t samplesPerIteration
              , const uint32_t updatesOnIteration );
  void Cancel();
  void Resize( const math::uvec2& size );
  void SetCameraParameters( const float fov
                            , const float focalLength
                            , const float aperture );
  void RotateCamera( const math::vec2& angles );
  
  void SetUpdateCallback( CallBackFunction callback );
  void SetFinishedCallback( CallBackFunction callback );

private:
  RayTracerImpl* mImpl;
};

}
