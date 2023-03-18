#include "RayTracer.h"
#include "RayTracerImpl.cuh"

namespace rt
{

RayTracer::RayTracer( const math::uvec2& imageSize
                      , const math::vec3& cameraPosition
                      , const math::vec2& cameraAngles
                      , const float fov
                      , const float focalLength
                      , const float aperture )
  : mImpl( new RayTracerImpl( imageSize
                              , cameraPosition
                              , cameraAngles
                              , fov
                              , focalLength
                              , aperture ) )
{}

RayTracer::~RayTracer()
{
  delete mImpl;
}

void RayTracer::Trace( const uint32_t iterationCount
                       , const uint32_t samplesPerIteration
                       , const uint32_t updateInterval )
{
  mImpl->Trace( iterationCount
                , samplesPerIteration
                , updateInterval );
}

void RayTracer::Stop()
{
  mImpl->Stop();
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

void RayTracer::RotateCamera( const math::vec2& angles )
{
  mImpl->RotateCamera( angles );
}

void RayTracer::UploadScene( const std::vector<float4>& hostData )
{
  mImpl->UploadScene( hostData );
}

void RayTracer::SetUpdateCallback( rt::CallBackFunction callback )
{
  mImpl->SetUpdateCallback( callback );
}

void RayTracer::SetFinishedCallback( rt::CallBackFunction callback )
{
  mImpl->SetFinishedCallback( callback );
}

}
