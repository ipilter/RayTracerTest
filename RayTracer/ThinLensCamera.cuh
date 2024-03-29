#pragma once

#include "Common\Sptr.h"
#include "Common\Math.h"

#include "Ray.cuh"
#include "DeviceUtils.cuh"
#include "Random.cuh"

namespace rt
{

class ThinLensCamera
{
public:
  __host__ __device__ ThinLensCamera( const math::vec3& position
                                      , const math::vec2& rotationAngles
                                      , const float fov // field of view [deg]
                                      , const float focalLength // focal length [mm?]
                                      , const float aperture ) // radius of lens [mm?]
    : mPosition( position )
    , mRotationAngles( rotationAngles )
    , mFov( glm::radians( fov ) )
    , mFocalLength( focalLength )
    , mAperture( aperture )
  {
    CalculateCameraTransformation();
  }

  __device__ Ray GetRay( const math::uvec2& pixel
                         , const math::uvec2& dimensions
                         , curandState_t& randomState ) const
  {
    // calculate real ray using physical camera properties
    // we assume the lens being positioned at camera position
    // https://drive.google.com/file/d/19mAlPb5YO-KDladvo3C8yzyrYmrudPJL/view?pli=1

    // first calculate the primary ray using a pinhole camera
    const Ray primary( PinHoleRay( pixel, dimensions ) );

    // random vec between 0.0 and aperture -> random pos on lens
    const math::vec3 randomOffset( math::vec3( random::UnifromOnDisk( randomState ) * mAperture, 0.0f ) );

    // focal point
    const math::vec3 focalPoint( Position() + mFocalLength * primary.direction() );

    // sample lens surface
    const math::vec3 randomLensPoint( Position() + randomOffset );

    // calculate final ray
    return Ray( randomLensPoint, focalPoint - randomLensPoint ); 
  }

  __host__ __device__ math::vec3 Position() const
  {
    return math::vec3( mCameraTransformation[3] );
  }

  __host__ __device__ math::vec3 Forward() const
  {
    return math::vec3( mCameraTransformation[0][2], mCameraTransformation[1][2], mCameraTransformation[2][2] );
  }

  __host__ __device__ math::vec3 Up() const
  {
    return math::vec3( mCameraTransformation[0][1], mCameraTransformation[1][1], mCameraTransformation[2][1] );
  }

  __host__ __device__ math::vec3 Right() const
  {
    return math::vec3( mCameraTransformation[0][0], mCameraTransformation[1][0], mCameraTransformation[2][0] );
  }

  __host__ __device__ float Fov() const
  {
    return mFov;
  }

  __host__ __device__ void Fov( const float fov )
  {
    mFov = glm::radians( fov );
  }

  __host__ __device__ float Aperture() const
  {
    return mAperture;
  }

  __host__ __device__ void Aperture( const float f )
  {
    mAperture = f;
  }

  __host__ __device__ float FocalLength() const
  {
    return mFocalLength;
  }

  __host__ __device__ void FocalLength( const float l )
  {
    mFocalLength = l;
  }

  __host__ void Rotate( const math::vec2& angles )
  {
    mRotationAngles += angles;
    CalculateCameraTransformation();
  }

private:
  __host__ __device__ Ray PinHoleRay( const math::uvec2& pixel
                                      , const math::uvec2& dimensions ) const
  {
    const float aspectRatio( dimensions.x / static_cast<float>( dimensions.y ) );
    const float halfHeight( glm::tan( mFov / 2.0f ) );

    const math::vec2 normalizedPixel( ( pixel.x + 0.5f ) / dimensions.x
                                      , ( pixel.y + 0.5f ) / dimensions.y );
    const math::vec2 cameraPixel( ( 2.0f * normalizedPixel.x - 1.0f ) * halfHeight * aspectRatio
                                  , ( 1.0f - 2.0f * normalizedPixel.y ) * halfHeight );

    const math::vec3 originCS( 0.0f, 0.0f, 0.0f );
    const math::vec3 pixelCS( cameraPixel, -1.0f );

    const math::vec3 originWS( math::vec3( mCameraTransformation * math::vec4( originCS, 1.0f ) ) );
    const math::vec3 pixelWS( math::vec3( mCameraTransformation * math::vec4( pixelCS, 1.0f ) ) );

    const math::vec3 direction( pixelWS - originWS );
    return Ray( originWS, direction );
  }

  __host__ __device__ void CalculateCameraTransformation()
  {
    // TODO store these in a permanent place (no static in device code)
    const math::vec3 xAxis( 1.0f, 0.0f, 0.0f );
    const math::vec3 yAxis( 0.0f, 1.0f, 0.0f );

    const math::quat qX( glm::angleAxis( mRotationAngles[0], xAxis ) );
    const math::quat qY( glm::angleAxis( mRotationAngles[1], yAxis ) ); 
    mCameraTransformation = glm::mat4_cast( qY * qX );
  }

private:
  math::vec3 mPosition;
  math::vec2 mRotationAngles;

  float mFov;         // field of view
  float mFocalLength; // focal point distance. The distance at which items in the image are in focus.
  float mAperture;    // radius of the aperture

  math::mat4 mCameraTransformation; // position and orientation
};

}
