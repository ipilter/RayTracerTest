#include "RayTracer.h"
#include "RenderKernel.cuh"
#include "ThinLensCamera.cuh"
#include "RenderData.cuh"

namespace rt
{

RayTracer::RayTracer()
{}

// TODO: use RenderData
void RayTracer::Trace( uint32_t* ptr
                       , const math::uvec2& size
                       , const uint32_t sampleCount
                       , const float fov
                       , const float focalLength
                       , const float aperture )
{
  rt::ThinLensCamera camera( math::vec3( 0.0f, 0.0f, 0.0f )
                             , math::vec3( 0.0f, 0.0f, 1.0f )
                             , math::vec3( 0.0f, 1.0f, 0.0f )
                             , fov, focalLength, aperture );

  // TODO: random states recreated every time !! Do it only if image size is changed
  rt::RenderData renderData( ptr, size, camera, sampleCount );
  RunRenderKernel( renderData );
}

void RayTracer::Resize( const math::uvec2& size )
{
  // recalculate random states if resize happens
}

}
