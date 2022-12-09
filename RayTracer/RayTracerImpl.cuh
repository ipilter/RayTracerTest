#include <curand_kernel.h>

#include "Common\Math.h"
#include <cstdint>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "ThinLensCamera.cuh"
#include "Random.cuh"

namespace rt
{

class RayTracerImpl
{
public:
  RayTracerImpl( const math::uvec2& pixelBufferSize );
  ~RayTracerImpl();

  void Trace( rt::color_t* pixelBufferPtr
              , const uint32_t sampleCount
              , const float fov
              , const float focalLength
              , const float aperture );

  void Resize( const math::uvec2& size );

private:
  cudaError_t RunRenderKernel( rt::color_t* pixelBufferPtr
                               , const math::uvec2& pixelBufferSize
                               , rt::ThinLensCamera& camera
                               , const uint32_t sampleCount
                               , curandState_t* randomStates );

  math::uvec2 mPixelBufferSize;
  curandState_t* mRandomStates;
};

}
