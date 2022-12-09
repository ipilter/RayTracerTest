#include "RayTracer.h"
#include "RayTracerImpl.cuh"

namespace rt
{

RayTracer::RayTracer( const math::uvec2& imageSize )
  : mImpl( new RayTracerImpl( imageSize ) )
{}

RayTracer::~RayTracer()
{
  delete mImpl;
}

void RayTracer::Trace( rt::color_t* pixelBufferPtr
                       , const uint32_t sampleCount
                       , const float fov
                       , const float focalLength
                       , const float aperture )
{
  mImpl->Trace( pixelBufferPtr, sampleCount, fov, focalLength, aperture );
}

void RayTracer::Resize( const math::uvec2& size )
{
  mImpl->Resize( size );
}

}
