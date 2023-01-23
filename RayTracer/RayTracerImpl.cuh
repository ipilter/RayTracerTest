#include "Common\Math.h"
#include <cstdint>
#include <functional>
#include <thread>

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
  RayTracerImpl( const math::uvec2& imageSize
                 , const math::vec3& cameraPosition
                 , const math::vec2& cameraAngles
                 , const float fov
                 , const float focalLength
                 , const float aperture );
  ~RayTracerImpl();

  void Trace( const uint32_t iterationCount
              , const uint32_t samplesPerIteration
              , const uint32_t updateInterval );
  void Cancel();
  void Resize( const math::uvec2& size );
  void SetCameraParameters( const float fov
                            , const float focalLength
                            , const float aperture );
  void RotateCamera( const math::vec2& angles );
  void SetUpdateCallback( CallBackFunction callback );
  void SetFinishedCallback( CallBackFunction callback );

private:
  __host__ cudaError_t RunTraceKernel( float* renderBuffer
                                       , const math::uvec2& bufferSize
                                       , const uint32_t channelCount
                                       , rt::ThinLensCamera& camera
                                       , const uint32_t sampleCount
                                       , curandState_t* randomStates );

  __host__ cudaError_t RunConverterKernel( const math::uvec2& bufferSize
                                           , const uint32_t channelCount
                                           , float*& renderBuffer
                                           , rt::color_t* imageBuffer );

  __host__ void TraceFunct( const uint32_t iterationCount
                            , const uint32_t samplesPerIteration
                            , const uint32_t updateInterval );

  void ReleaseBuffers();

private:
  math::uvec2 mBufferSize;
  curandState_t* mRandomStates;
  float* mRenderBuffer;
  std::unique_ptr<rt::ThinLensCamera> mCamera;

  std::atomic<bool> mCancelled;
  CallBackFunction mUpdateCallback;
  CallBackFunction mFinishedCallback;
  std::thread mThread;
};

}
