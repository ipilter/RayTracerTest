#pragma once

#include <cstdint>
#include <memory>

namespace rt
{

using Channel = uint8_t;
using Color = uint32_t;
using ColorConstPtr = const Color* const;

// BGRA -> r|g|b|a
inline rt::Channel GetComponent( const Color& color, const uint32_t& idx )
{
  static const uint32_t params[][2]{ {0x00FF0000, 16}, {0x0000FF00, 8}, {0x000000FF, 0}, {0xFF000000, 24} };
  return static_cast<rt::Channel>( ( color & params[idx][0] ) >> params[idx][1] );
}

// r|g|b|a -> BGRA
inline Color GetColor( const Channel r = 0, const Channel g = 0, const Channel b = 0, const Channel a = 255 )
{
  return ( b << 0 ) | ( g << 8 ) | ( r << 16 ) | ( a << 24 );
}

}
