#pragma once

#include "Common\Math.h"
#include "Common\Sptr.h"
#include "Common\Color.h"

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

  rt::color_t* MapPboBuffer();
  void UnMapPboBuffer();

private:
  uint32_t mId;
};

}
