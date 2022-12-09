#pragma once

#include <cstdint>

namespace rt
{

using channel_t = uint8_t;
using color_t = uint32_t;

// BGRA -> r|g|b|a
inline rt::channel_t GetComponent( const color_t& color, const uint32_t& idx )
{
  static const uint32_t params[][2]{ {0x00FF0000, 16}, {0x0000FF00, 8}, {0x000000FF, 0}, {0xFF000000, 24} };
  return static_cast<rt::channel_t>( ( color & params[idx][0] ) >> params[idx][1] );
}

// r|g|b|a -> BGRA
inline color_t Color( const channel_t r = 0, const channel_t g = 0, const channel_t b = 0, const channel_t a = 255 )
{
  return ( b << 0 ) | ( g << 8 ) | ( r << 16 ) | ( a << 24 );
}

}
