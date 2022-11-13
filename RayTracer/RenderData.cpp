#include "RenderData.h"

namespace rt
{

RenderData::RenderData( uint32_t* ptr, const math::uvec2& size )
  : mPixelBuffer( ptr )
  , mDimensions( size )
{}

RenderData::~RenderData()
{}

uint32_t* RenderData::PixelBuffer()
{
  return mPixelBuffer;
}

const math::uvec2& RenderData::Dimensions() const
{
  return mDimensions;
}

}
