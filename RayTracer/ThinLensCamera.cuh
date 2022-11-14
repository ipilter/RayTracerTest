#pragma once

#include "Common\Sptr.h"
#include "Common\Math.h"
#include "Ray.cuh"
#include "Utils.cuh"

namespace rt
{

class ThinLensCamera
{
public:
  __host__ __device__ ThinLensCamera( const math::vec3& position
                                      , const math::vec3& target
                                      , const math::vec3& upGuide
                                      , const float fov // deg
                                      , const float focalLength // m?
                                      , const float aperture )
    : mCameraTransformation( calculateCameraTransformation( position, target, upGuide ) )
    , mFov( glm::radians( fov ) )
    , mFocalLength( focalLength )
    , mAperture( aperture )
  {}

  __host__ __device__ Ray getRay( const math::uvec2& pixel, const math::uvec2& dimensions ) const
  {
    // first calculate the primary ray of the pinhole camera
    const Ray primary = PinHoleRay( pixel, dimensions );
    return primary;

    // calculate real ray using physical camera properties
    // we assume the lens being positioned at camera position

    // https://drive.google.com/file/d/19mAlPb5YO-KDladvo3C8yzyrYmrudPJL/view?pli=1
    
    // random vec between 0.0 and aperture -> random pos on lens
    //const math::vec3 randomOffset( math::vec3( rt::randomOnCircle() * mAperture, 0.0 ) );
    //const math::vec3 focalPoint( position() + mFocalLength * primary.direction() );

    //const math::vec3 randomLensPoint( position() + randomOffset );
    //const math::vec3 direction( focalPoint - randomLensPoint );
    //return Ray( randomLensPoint, direction ); 
  }

  __host__ __device__ math::vec3 position() const
  {
    return math::vec3( mCameraTransformation[3] );
  }

  __host__ __device__ math::vec3 forward() const
  {
    return math::vec3( mCameraTransformation[0][2], mCameraTransformation[1][2], mCameraTransformation[2][2] );
  }

  __host__ __device__ math::vec3 up() const
  {
    return math::vec3( mCameraTransformation[0][1], mCameraTransformation[1][1], mCameraTransformation[2][1] );
  }

  __host__ __device__ math::vec3 right() const
  {
    return math::vec3( mCameraTransformation[0][0], mCameraTransformation[1][0], mCameraTransformation[2][0] );
  }

  __host__ __device__ float fov() const
  {
    return mFov;
  }

  __host__ __device__ void fov( const float fov )
  {
    mFov = glm::radians( fov );
  }

  __host__ __device__ float aperture() const
  {
    return mAperture;
  }

  __host__ __device__ void aperture( const float f )
  {
    mAperture = f;
  }

  __host__ __device__ float focalLength() const
  {
    return mFocalLength;
  }

  __host__ __device__ void focalLength( const float l )
  {
    mFocalLength = l;
  }

private:
  __host__ __device__ Ray PinHoleRay( const math::uvec2& pixel, const math::uvec2& dimensions ) const
  {
    const float max( static_cast<float>( glm::min( dimensions.x, dimensions.y ) ) );
    const float min( static_cast<float>( glm::max( dimensions.x, dimensions.y ) ) );
    const float aspectRatio( max / min );
    const float halfHeight( glm::tan( mFov / 2.0f ) );

    const math::vec2 normalizedPixel( ( pixel.x + 0.5f ) / dimensions.x
                                      , ( pixel.y + 0.5f ) / dimensions.y );
    const math::vec2 cameraPixel( ( 2.0f * normalizedPixel.x - 1.0f ) * halfHeight * aspectRatio
                                  , ( 1.0f - 2.0f * normalizedPixel.y ) * halfHeight );

    const math::vec3 originCS( 0.0, 0.0, 0.0 );
    const math::vec3 pixelCS( cameraPixel, -1.0 );

    const math::vec3 originWS( math::vec3( mCameraTransformation * math::vec4( originCS, 1.0 ) ) );
    const math::vec3 pixelWS( math::vec3( mCameraTransformation * math::vec4( pixelCS, 1.0 ) ) );

    const math::vec3 direction( pixelWS - originWS );
    return Ray( originWS, direction );
  }

  __host__ __device__ static  math::mat4 calculateCameraTransformation( const math::vec3& position
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

  float mFov;
  float mFocalLength; // focal point distance. The distance at which items in the image are in focus.
  float mAperture; // radius of the aperture
  math::mat4 mCameraTransformation;
};

}
