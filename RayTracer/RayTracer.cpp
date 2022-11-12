#include "RayTracer.h"
#include "kernel.cuh"

namespace rt
{

RayTracer::RayTracer()
{ }

void RayTracer::Trace( rt::RenderData& renderData )
{
  RunUpdateTextureKernel( renderData );
}

}
