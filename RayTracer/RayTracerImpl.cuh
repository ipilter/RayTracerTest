#include "Common\Math.h"
#include <cstdint>
#include <functional>
#include <thread>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>

#include "ThinLensCamera.cuh"
#include "Random.cuh"
#include "RaytracerCallback.h"

namespace rt
{

class RayTracerImpl
{
public:
  

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
  void Stop();
  void Resize( const math::uvec2& size );
  void SetCameraParameters( const float fov
                            , const float focalLength
                            , const float aperture );
  void RotateCamera( const math::vec2& angles );
  void SetUpdateCallback( rt::CallBackFunction callback );
  void SetFinishedCallback( rt::CallBackFunction callback );

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
                                           , rt::Color* imageBuffer );

  __host__ void TraceFunct( const uint32_t iterationCount
                            , const uint32_t samplesPerIteration
                            , const uint32_t updateInterval );

  void ReleaseBuffers();

  static unsigned ChannelCount()
  {
    return 4;
  }

private:
  math::uvec2 mBufferSize;
  float* mRenderBuffer; // TODO no raw ptr
  rt::Color* mImageBuffer; // TODO no raw ptr

  curandState_t* mRandomStates; // TODO no raw ptr
  std::unique_ptr<rt::ThinLensCamera> mCamera;

  std::atomic<bool> mStopped;
  rt::CallBackFunction mUpdateCallback;
  rt::CallBackFunction mFinishedCallback;
  std::thread mThread;
};

}
