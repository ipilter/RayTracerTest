#pragma once

#include <vector>
#include <string>
#include <stdexcept>
#include <fstream>

#include "Math.h"

namespace rt
{

// TODO: not compatible with PBO pixel data! 
// Need conversaion as PBO stores an rgba pixel as an uint32_t and Ppm stores rgb pixels as 3 char!
class Ppm
{
public:
  struct Pixel
  {
    using ChannelType = unsigned char;
    ChannelType  mRed, mGreen, mBlue;

    Pixel( const ChannelType r = 0, const ChannelType g = 0, const ChannelType b = 0 );

    static ChannelType maxColor();

    Pixel& operator += ( const Pixel& rhs );
    Pixel operator + ( const Pixel& rhs );
    Pixel& operator -= ( const Pixel& rhs );
    Pixel operator - ( const Pixel& rhs );
    Pixel operator * ( const float s ) const;
  };

  using ImageData = std::vector<Pixel>;
  using SizeType = uint32_t;
  using sptr = std::shared_ptr<Ppm>;

  /*
  * Creator method
  */
  static sptr create( const math::uvec2 & size, const Pixel& color = Pixel());

  /*
  * Constructs a new image with given width, height and default pixel color
  */
  Ppm( const math::uvec2& size, const Pixel& color = Pixel() );

public:
  /*
  * Sets the pixel at the given coordinates
  */
  void setPixel( const math::uvec2& pos, const Pixel& p );

  /*
  * Returns the pixel at the given coordinates
  */
  Pixel getPixel( const math::uvec2& pos ) const;

  /*
  * Fills image with a given color
  */
  void fill( const Pixel& color = Pixel() );

public:
  /*
  * Returns the image size in pixels (WxH)
  */
  const math::uvec2& size() const;

  /*
  * Returns the image size in bytes
  */
  uint64_t bytes() const;

  /*
  * Returns the raw memory address of the first Pixel
  */
  Pixel* memory();

public:
  /*
  * Writes the current image to the given file path
  */
  void save( const std::string& file ) const;

private:
  math::uvec2 mSize;
  ImageData mImageData;
};
}
