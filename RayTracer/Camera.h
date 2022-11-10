#pragma once

#include <vector>

#include "..\Sptr.h"
#include "..\Math.h"
#include "Ray.h"

namespace rt
{
class ICamera
{
public:
  using sptr = std::shared_ptr<ICamera>;

public:
  virtual ~ICamera() = 0;

  virtual math::vec3 position() const = 0;
  virtual math::vec3 forward() const = 0;
  virtual math::vec3 up() const = 0;
  virtual math::vec3 right() const = 0;

  virtual void getRays( const math::uvec2& pixel
                       , uint32_t count
                       , std::vector<Ray>& rays ) const = 0;
};

class ACamera : public ICamera
{
public:
  ACamera( const math::vec3& position
           , const math::vec3& target
           , const math::vec3& upGuide );

public:
  virtual math::vec3 position() const;
  virtual math::vec3 forward() const;
  virtual math::vec3 up() const;
  virtual math::vec3 right() const;

private:
  static math::mat4 calculateCameraTransformation( const math::vec3& position
                                                   , const math::vec3& target
                                                   , const math::vec3& upGuide );

protected:
  math::mat4 mCameraTransformation;
};

class PinholeCamera : public ACamera
{
public:
  PinholeCamera( const math::vec3& position
                 , const math::vec3& target
                 , const math::vec3& upGuide
                 , const float fov ); // radians

  virtual void getRays( const math::uvec2& pixel
                        , uint32_t count
                        , std::vector<Ray>& rays ) const;

  float fov() const;
  void fov( const float theta );

private:
  float mFov;
};

class ThinLensCamera : public PinholeCamera
{
public:
  ThinLensCamera( const math::vec3& position
                  , const math::vec3& target
                  , const math::vec3& upGuide
                  , const float fov // radians
                  , const float focalLength // m?
                  , const float aperture );  // m?

  virtual void getRays( const math::uvec2& pixel
                        , uint32_t count
                        , std::vector<Ray>& rays ) const;

  float aperture() const;
  void aperture( const float a );

  float focalLength() const;
  void focalLength( const float l );

private:
  float mFocalLength; // focal point distance. The distance at which items in the image are in focus.
  float mAperture; // radius of the aperture
};

}
