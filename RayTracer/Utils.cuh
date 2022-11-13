#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace utils
{
__device__ uint8_t Component( const uint32_t& color, const uint32_t& idx )
{
  switch ( idx )
  {
    case 0:
      return static_cast<uint8_t>( ( color & 0x000000FF ) >> 0 );
    case 1:
      return static_cast<uint8_t>( ( color & 0x0000FF00 ) >> 8 );
    case 2:
      return static_cast<uint8_t>( ( color & 0x00FF0000 ) >> 16 );
    case 3:
      return static_cast<uint8_t>( ( color & 0xFF000000 ) >> 24 );
    default:
      return 0;
  }
}

__device__ uint32_t Color( const uint8_t r = 0, const uint8_t g = 0, const uint8_t b = 0, const uint8_t a = 255 )
{
  return ( r << 0 ) | ( g << 8 ) | ( b << 16 ) | ( a << 24 );
}

} // ::utils
