#include "RayTracer.h"
#include "RayTracerImpl.cuh"

namespace rt
{

RayTracer::RayTracer( const math::uvec2& imageSize
                      , const float fov
                      , const float focalLength
                      , const float aperture )
  : mImpl( new RayTracerImpl( imageSize
                              , fov
                              , focalLength
                              , aperture ) )
{}

RayTracer::~RayTracer()
{
  delete mImpl;
}

void RayTracer::Trace( rt::color_t* pixelBufferPtr, const uint32_t sampleCount )
{
  mImpl->Trace( pixelBufferPtr, sampleCount );
}

void RayTracer::Resize( const math::uvec2& size )
{
  mImpl->Resize( size );
}

void RayTracer::SetCameraParameters( const float fov
                                     , const float focalLength
                                     , const float aperture )
{
  mImpl->SetCameraParameters( fov, focalLength, aperture );
}

void RayTracer::RotateCamera( const math::uvec2& angles )
{
  mImpl->RotateCamera( angles );
}

}
