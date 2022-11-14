#include "RayTracer.h"
#include "kernel.cuh"
#include "ThinLensCamera.cuh"
#include "RenderData.cuh"

namespace rt
{

RayTracer::RayTracer()
{}

void RayTracer::Trace( uint32_t* ptr, const math::uvec2& size )
{
  rt::ThinLensCamera camera( math::vec3( 0.0, 0.0, 0.0 ), math::vec3( 0.0, 0.0, 1.0 ), math::vec3( 0.0, 1.0, 0.0 ), 60.0f, 3.0f, 1.0f );
  rt::RenderData renderData( ptr, size, camera );
  RunRenderKernel( renderData );
}

}
