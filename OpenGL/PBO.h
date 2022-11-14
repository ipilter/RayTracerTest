#pragma once

#include "Common\Math.h"
#include "Common\Sptr.h"

namespace gl
{

// pixel buffer object
class PBO : public virtual ISptr<PBO>
{
public:
  PBO();
  ~PBO();

  void Allocate( const uint32_t byteCount );

  void Bind();
  void Unbind();

  const uint32_t& Id() const;

  uint32_t* MapPboBuffer();
  void UnMapPboBuffer();

private:
  uint32_t mId;
};

}
