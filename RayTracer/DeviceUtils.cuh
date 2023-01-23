#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "Common/Math.h"
#include "Common/Color.h"

namespace utils
{

// BGRA -> r|g|b|a
inline __device__ rt::Channel GetComponent( const rt::Color& color, const uint32_t& idx )
{
  static const uint32_t params[][2]{ {0x00FF0000, 16}, {0x0000FF00, 8}, {0x000000FF, 0}, {0xFF000000, 24} };
  return static_cast<rt::Channel>( ( color & params[idx][0] ) >> params[idx][1] );
}

// r|g|b|a -> BGRA
inline __device__ rt::Color GetColor( const rt::Channel r = 0, const rt::Channel g = 0, const rt::Channel b = 0, const rt::Channel a = 255 )
{
  return ( b << 0 ) | ( g << 8 ) | ( r << 16 ) | ( a << 24 );
}

} // ::utils
