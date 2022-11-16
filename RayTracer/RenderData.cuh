#pragma once
#include "Common\Math.h"
#include "ThinLensCamera.cuh"
#include "Random.cuh"

namespace rt
{

struct RenderData
{
  RenderData( uint32_t* ptr
              , const math::uvec2& size
              , const ThinLensCamera& camera
              , const uint32_t sampleCount )
    : mPixelBuffer( ptr )
    , mDimensions( size )
    , mCamera( camera )
    , mRandom( size )
    , mSampleCount( sampleCount )
  {}

  uint32_t* mPixelBuffer;
  math::uvec2 mDimensions;
  ThinLensCamera mCamera;
  Random mRandom;
  uint32_t mSampleCount;
};

}
