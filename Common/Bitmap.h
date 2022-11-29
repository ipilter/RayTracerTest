#pragma once

#include <string>
#include <fstream>

#include "HostUtils.h"
#include "Math.h"

class Bitmap
{
public:
  using channel_t = uint8_t;
  using color_t = uint32_t;

public:
  Bitmap( const math::uvec2& size, const color_t fillColor = 0x00000000 )
    : mSize( size )
    , mImageData( new color_t[static_cast<size_t>( size.x ) * size.y] )
  {
    std::fill( mImageData, mImageData + static_cast<size_t>( size.x ) * size.y, fillColor );
  }

  ~Bitmap()
  {
    delete[] mImageData;
  }

  void SetPixel( const uint32_t x, const uint32_t y, const color_t& color )
  {
    const uint32_t offset( y * mSize.x + x );
    mImageData[offset] = color;
  }

  color_t GetPixel( const uint32_t x, const uint32_t y ) const
  {
    const uint32_t offset( y * mSize.x + x );
    return mImageData[offset];
  }

  const glm::uvec2& Size() const
  {
    return mSize;
  }

  void Write( const std::string& path ) const
  {
    ImageHeader imageHeader;
    imageHeader.size = sizeof( ImageHeader ) + sizeof( ColorHeader );
    imageHeader.width = mSize.x;
    imageHeader.height = mSize.y * -1;

    FileHeader fileHeader;
    fileHeader.dataOffset = sizeof( FileHeader ) + sizeof( ImageHeader ) + sizeof( ColorHeader );
    fileHeader.fileSize = fileHeader.dataOffset + sizeof( color_t ) * mSize.x * mSize.y;

    ColorHeader colorHeader;
    std::ofstream ostream( path, std::ios::out | std::ios::binary );
    util::Write_t( ostream, fileHeader );
    util::Write_t( ostream, imageHeader );
    util::Write_t( ostream, colorHeader );
    ostream.write( reinterpret_cast<const char*>( mImageData ), sizeof( color_t ) * mSize.x * mSize.y );
  }

public:
  static channel_t GetComponent( const color_t& color, const uint32_t& idx )
  {
    static const std::pair<uint32_t, uint32_t> params[]{ {0x00FF0000, 16}, {0x0000FF00, 8}, {0x000000FF, 0}, {0xFF000000, 24} };
    return static_cast<channel_t>( ( color & params[idx].first ) >> params[idx].second );
  }

  static color_t GetColor( const channel_t r = 0, const channel_t g = 0, const channel_t b = 0, const channel_t a = 255 )
  {
    return ( b << 0 ) | ( g << 8 ) | ( r << 16 ) | ( a << 24 );
  }

private:
  glm::uvec2 mSize;
  color_t* mImageData;

private:
#pragma pack(push, 1)
  struct FileHeader
  {
    uint16_t signature{ 0x4D42 }; // File type always BM which is 0x4D42
    uint32_t fileSize{ 0 };       // Size of the file (in bytes)
    uint16_t reserved[2]{ 0 };    // Reserved, always 0
    uint32_t dataOffset{ 0 };     // Start position of pixel data (bytes from the beginning of the file)
  };

  struct ImageHeader
  {
    uint32_t size{ 0 };          // Size of this header (in bytes)
    int32_t width{ 0 };          // width of bitmap in pixels
    int32_t height{ 0 };         // height of bitmap in pixels + -> bottom up, - -> top-down
    uint16_t planes{ 1 };        // No. of planes for the target device
    uint16_t bpp{ 32 };          // No. of bits per pixel
    uint32_t compression{ 3 };   // 3 - uncompressed 32bit.
    uint32_t imageSize{ 0 };
    int32_t widthPpm{ 0 };
    int32_t heightPpm{ 0 };
    uint32_t usedColors{ 0 };
    uint32_t importantColors{ 0 };
  };

  struct ColorHeader
  {
    uint32_t red_mask{ 0x00FF0000 };         // Bit mask for the red channel
    uint32_t green_mask{ 0x0000FF00 };       // Bit mask for the green channel
    uint32_t blue_mask{ 0x000000FF };        // Bit mask for the blue channel
    uint32_t alpha_mask{ 0xFF000000 };       // Bit mask for the alpha channel
    uint32_t color_space_type{ 0x73524742 }; // Default "sRGB" (0x73524742)
    uint32_t unused[16]{ 0 };                // Unused data for sRGB color space
  };
#pragma pack(pop)
};
