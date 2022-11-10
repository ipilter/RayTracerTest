#include "RayTracer.h"
#include "../kernel.cuh"

namespace rt
{

RayTracer::RayTracer()
{}

void RayTracer::Trace( gl::PixelBufferObject::sptr& pbo, const math::uvec2& pixelsSize ) // TODO render data instead these
{
  pbo->MapCudaResource();
  uint32_t* devicePixelPtr( pbo->GetCudaMappedPointer() ); // TODO wrapper to be exception safe

  RunUpdateTextureKernel( devicePixelPtr, pixelsSize.x, pixelsSize.y ); // TODO render data instead these

  pbo->UnmapCudaResource();  // TODO wrapper to be exception safe
}

}
