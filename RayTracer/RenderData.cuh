#pragma once
#include "Common\Math.h"
#include "ThinLensCamera.cuh"

namespace rt
{

struct RenderData
{
  RenderData( uint32_t* ptr, const math::uvec2& size, const ThinLensCamera& camera )
    : mPixelBuffer( ptr )
    , mDimensions( size )
    , mCamera( camera )
  {}

  uint32_t* mPixelBuffer;
  math::uvec2 mDimensions;
  ThinLensCamera mCamera;
};

}
