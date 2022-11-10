#include "Camera.h"

namespace rt
{
/// /////////////////////////////////
/// ICamera
/// /////////////////////////////////
ICamera::~ICamera()
{ }

/// /////////////////////////////////
/// ACamera
/// /////////////////////////////////
ACamera::ACamera( const math::vec3& position
                  , const math::vec3& target
                  , const math::vec3& upGuide )
  : mCameraTransformation( calculateCameraTransformation( position, target, upGuide ) )
{}

math::vec3 ACamera::position() const
{
  return math::vec3( mCameraTransformation[3] );
}

math::vec3 ACamera::right() const
{
  return math::vec3( mCameraTransformation[0][0], mCameraTransformation[1][0], mCameraTransformation[2][0] );
}

math::vec3 ACamera::up() const
{
  return math::vec3( mCameraTransformation[0][1], mCameraTransformation[1][1], mCameraTransformation[2][1] );
}

math::vec3 ACamera::forward() const
{
  return math::vec3( mCameraTransformation[0][2], mCameraTransformation[1][2], mCameraTransformation[2][2] );
}

math::mat4 ACamera::calculateCameraTransformation( const math::vec3& position
                                                   , const math::vec3& target
                                                   , const math::vec3& upGuide )
{
  const math::vec3 forward = glm::normalize( position - target );
  const math::vec3 right = glm::normalize( glm::cross( upGuide, forward ) );
  const math::vec3 up = glm::cross( forward, right );

  return math::mat4( right.x, up.x, forward.x, 0.0
                     , right.y, up.y, forward.y, 0.0
                     , right.z, up.z, forward.z, 0.0
                     , position.x, position.y, position.z, 1.0 );
}

/// /////////////////////////////////
/// PinholeCamera
/// /////////////////////////////////
PinholeCamera::PinholeCamera( const math::vec3& position
                              , const math::vec3& target
                              , const math::vec3& upGuide
                              , const float fov )
  : ACamera( position, target, upGuide )
  , mFov( fov )
{}

void PinholeCamera::getRays( const math::uvec2& pixel
                             , uint32_t /*count*/
                             , std::vector<Ray>& rays ) const
{
  //const double max( std::min( mFramebuffer->size().x, mFramebuffer->size().y ) );
  //const double min( std::max( mFramebuffer->size().x, mFramebuffer->size().y ) );
  //const double aspectRatio( max / min );
  //const double halfHeight( std::tan( mFov / 2.0 ) );

  //const math::vec2 normalizedPixel( ( pixel.x + 0.5 ) / mFramebuffer->size().x
  //                            , ( pixel.y + 0.5 ) / mFramebuffer->size().y );
  //const math::vec2 cameraPixel( ( 2 * normalizedPixel.x - 1 ) * halfHeight * aspectRatio
  //                        , ( 1 - 2 * normalizedPixel.y ) * halfHeight );

  //const math::vec3 originCS( 0.0, 0.0, 0.0 );
  //const math::vec3 pixelCS( cameraPixel, -1.0 );

  //const math::vec3 originWS( math::vec3( mCameraTransformation * math::vec4( originCS, 1.0 ) ) );
  //const math::vec3 pixelWS( math::vec3( mCameraTransformation * math::vec4( pixelCS, 1.0 ) ) );

  //const math::vec3 direction( pixelWS - originWS );
  //rays.push_back( Ray( originWS, direction ) );
}

float PinholeCamera::fov() const
{
  return mFov;
}

void PinholeCamera::fov( const float fov )
{
  mFov = math::rad( fov );
}

/// /////////////////////////////////
/// ThinLensCamera
/// /////////////////////////////////
ThinLensCamera::ThinLensCamera( const math::vec3& position
                                , const math::vec3& target
                                , const math::vec3& upGuide
                                , const float fov
                                , const float focalLength
                                , const float aperture )
  : PinholeCamera( position, target, upGuide, fov )
  , mFocalLength( focalLength )
  , mAperture( aperture )
{ }

void ThinLensCamera::getRays( const math::uvec2& pixel
                              , uint32_t count
                              , std::vector<Ray>& rays ) const
{
  //// we assume the lens being positioned at camera position

  //// https://drive.google.com/file/d/19mAlPb5YO-KDladvo3C8yzyrYmrudPJL/view?pli=1
  //std::vector<Ray> primaryRays;
  //PinholeCamera::getRays( pixel, 1, primaryRays );
  //
  //rays.reserve( count );
  //for ( uint32_t i = 0; i < count; ++i )
  //{
  //  // random vec between 0.0 and aperture -> random pos on lens
  //  const math::vec3 randomOffset( math::vec3( rt::randomOnCircle() * mAperture, 0.0 ) );
  //  const math::vec3 focalPoint( position() + mFocalLength * primaryRays[0].direction() );

  //  const math::vec3 randomLensPoint( position() + randomOffset );
  //  const math::vec3 direction( focalPoint - randomLensPoint );
  //  rays.push_back( Ray( randomLensPoint, direction ) );
  //}
}

float ThinLensCamera::aperture() const
{
  return mAperture;
}

void ThinLensCamera::aperture( const float a )
{
  mAperture = a;
}

float ThinLensCamera::focalLength() const
{
  return mFocalLength;
}

void ThinLensCamera::focalLength( const float l )
{
  mFocalLength = l;
}

}
