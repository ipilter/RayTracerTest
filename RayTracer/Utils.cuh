#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "Common/Math.h"

namespace utils
{
inline __device__ uint8_t Component( const uint32_t& color, const uint32_t& idx )
{
  static const uint32_t params[][2] = { {0x000000FF, 0}, {0x0000FF00, 8},{0x00FF0000, 16}, {0xFF000000, 24} };   // RGBA
  return static_cast<uint8_t>( ( color & params[idx][0] ) >> params[idx][1] );
}

inline __device__ uint32_t Color( const uint8_t r = 0, const uint8_t g = 0, const uint8_t b = 0, const uint8_t a = 255 )
{
  return ( b << 0 ) | ( g << 8 ) | ( r << 16 ) | ( a << 24 );   // BGRA
}

} // ::utils
