#pragma once

#include "Common\Math.h"
#include "Common\Sptr.h"

namespace rt
{

class RayTracer : public ISptr<RayTracer>
{
public:
  RayTracer();
  void Trace( uint32_t* ptr, const math::uvec2& size );

private:
};

}
