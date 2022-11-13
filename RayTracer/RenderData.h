#pragma once
#include "Common\Math.h"

namespace rt
{

struct RenderData
{
  RenderData( uint32_t* ptr, const math::uvec2& size );
  ~RenderData();

  uint32_t* PixelBuffer();
  const math::uvec2& Dimensions() const;

private:
  uint32_t* mPixelBuffer;
  math::uvec2 mDimensions;
};

}
