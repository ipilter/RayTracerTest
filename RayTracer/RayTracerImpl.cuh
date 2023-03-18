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
  void UploadScene( const std::vector<float4>& hostData );

  void SetUpdateCallback( rt::CallBackFunction callback );
  void SetFinishedCallback( rt::CallBackFunction callback );

private:
  __host__ cudaError_t RunTraceKernel( const uint32_t sampleCount );
  __host__ cudaError_t RunConverterKernel();

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
  uint32_t* mSampleCountBuffer; // TODO no raw ptr
  rt::Color* mImageBuffer; // TODO no raw ptr

  curandState_t* mRandomStates; // TODO no raw ptr
  std::unique_ptr<rt::ThinLensCamera> mCamera;

  // Scene
  cudaArray_t mSceneTrianglesArray;
  cudaTextureObject_t mSceneTrianglesTextureObject;
  uint32_t mNumberOfTriangles;

  std::atomic<bool> mStopped;
  rt::CallBackFunction mUpdateCallback;
  rt::CallBackFunction mFinishedCallback;
  std::thread mThread;
};

}
