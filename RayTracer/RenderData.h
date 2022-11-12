#pragma once
#include "Common\Math.h"

namespace rt
{

struct RenderData
{
  RenderData();
  ~RenderData();

  uint32_t* mPixelBuffer;
  math::uvec2 mDimensions;
};

}
