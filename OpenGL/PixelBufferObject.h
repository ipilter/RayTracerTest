#pragma once

#include <driver_types.h>

#include "..\Math.h"
#include "..\Sptr.h"

namespace gl
{

// pixel buffer object with cuda resource
class PixelBufferObject : public virtual ISptr<PixelBufferObject>
{
public:
  PixelBufferObject();
  ~PixelBufferObject();

  void Allocate( uint32_t byteCount );

  void Bind();
  void Unbind();

  uint32_t* MapPboBuffer();
  void UnmapPboBuffer();

  // cuda
  void RegisterCudaResource();
  void MapCudaResource();
  void UnmapCudaResource();
  uint32_t* GetCudaMappedPointer();

private:
  uint32_t mPboId = 0;
  cudaGraphicsResource_t mCudaResource = 0;
  bool mBound = false;
};

}
