#pragma once

#include "Common\Math.h"
#include "Common\Sptr.h"

// pixel buffer object
class PBO : public virtual ISptr<PBO>
{
public:
  PBO();
  ~PBO();

  void Allocate( uint32_t byteCount );

  uint32_t Id() const;

  void Bind();
  void Unbind();

  uint32_t* MapBuffer();
  void UnmapBuffer();

private:
  uint32_t mId = 0;
};
