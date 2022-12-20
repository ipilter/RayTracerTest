#include "Common\Math.h"
#include <cstdint>
#include <functional>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>

#include "ThinLensCamera.cuh"
#include "Random.cuh"

namespace rt
{

class RayTracerImpl
{
public:
  using CallBackFunction = std::function<void()>;

public:
  RayTracerImpl( const math::uvec2& pixelBufferSize
                 , const math::vec3& cameraPosition
                 , const math::vec2& cameraAngles
                 , const float fov
                 , const float focalLength
                 , const float aperture );
  ~RayTracerImpl();

  void Trace( rt::color_t* pixelBufferPtr, const uint32_t sampleCount );
  void Resize( const math::uvec2& size );
  void SetCameraParameters( const float fov
                            , const float focalLength
                            , const float aperture );
  void RotateCamera( const math::vec2& angles );

  void SetDoneCallback( CallBackFunction callback );

private:
  cudaError_t RunRenderKernel( rt::color_t* pixelBufferPtr
                               , const math::uvec2& pixelBufferSize
                               , rt::ThinLensCamera& camera
                               , const uint32_t sampleCount
                               , curandState_t* randomStates );

  math::uvec2 mPixelBufferSize;
  curandState_t* mRandomStates;
  std::unique_ptr<rt::ThinLensCamera> mCamera;

  CallBackFunction mDoneCallback;
};

}
