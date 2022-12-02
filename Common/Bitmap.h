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
    , mImageData( static_cast<size_t>( size.x )* size.y, fillColor )
  {}

  Bitmap( const math::uvec2& size, std::vector<color_t>& pixelArray )
    : mSize( size )
    , mImageData()
  {
    std::swap( mImageData, pixelArray );
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
    std::ofstream ostream( path, std::ios::out | std::ios::binary );
    if ( !ostream.good() )
    {
      throw std::runtime_error( std::string( "cannot save file to path " ) + path );
    }

    Header header( mSize.x, mSize.y );
    header.Write( ostream );
    ostream.write( reinterpret_cast<const char*>( &mImageData.front() ), sizeof( color_t ) * mSize.x * mSize.y );
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
  std::vector<color_t> mImageData;

private:
  class Header
  {
  public:
    Header( const int32_t w, const int32_t h )
      : mFileHeader( w, h )
      , mImageHeader( w, h )
    {}

    void Write( std::ofstream& ostr )
    {
      util::Write_t( ostr, *this );
    }

  private:
#pragma pack(push, 1)
    class FileHeader
    {
    public:
      FileHeader( const int32_t w, const int32_t h )
        : mFileSize( sizeof( FileHeader ) + sizeof( ImageHeader ) + sizeof( ColorHeader ) + sizeof( color_t ) * w * h )
        , mDataOffset( sizeof( FileHeader ) + sizeof( ImageHeader ) + sizeof( ColorHeader ) )
      {}

    private:
      uint16_t mSignature{ 0x4D42 };           // File type (BM)
      uint32_t mFileSize{ 0 };                 // Size of the file (bytes)
      uint16_t mReserved[2]{ 0 };              // Reserved
      uint32_t mDataOffset{ 0 };               // Start position of pixel data (bytes from the beginning of the file)
    };
    class ImageHeader
    {
    public:
      ImageHeader( const int32_t w, const int32_t h )
        : mSize( sizeof( ImageHeader ) + sizeof( ColorHeader ) )
        , mWidth( w )
        , mHeight( -h )
      {}

    private:
      uint32_t mSize{ 0 };                     // Size of this header (bytes)
      int32_t  mWidth{ 0 };                    // Width of bitmap (pixels)
      int32_t  mHeight{ 0 };                   // Height of bitmap (pixels, > 0 -> bottom up, < 0 -> top-down)
      uint16_t mPlanes{ 1 };                   // Number of planes for the target device
      uint16_t mBpp{ 32 };                     // Number of bits per pixel
      uint32_t mCompression{ 3 };              // 3 for uncompressed 32bit
      uint32_t mImageSize{ 0 };                // 0 by default
      int32_t  mWidthPpm{ 0 };                 // 0 by default
      int32_t  mHeightPpm{ 0 };                // 0 by default
      uint32_t mUsedColors{ 0 };               // 0 by default
    };
    class ColorHeader
    {
      uint32_t mImportantColors{ 0 };          // 0 by default
      uint32_t mRedMask{ 0x00FF0000 };         // Bit mask for the red channel
      uint32_t mGreenMask{ 0x0000FF00 };       // Bit mask for the green channel
      uint32_t mBlueMask{ 0x000000FF };        // Bit mask for the blue channel
      uint32_t mAlphaMask{ 0xFF000000 };       // Bit mask for the alpha channel
      uint32_t mColorSpaceType{ 0x73524742 };  // Default "sRGB" (0x73524742)
      uint32_t mUnused[16]{ 0 };               // Unused data for sRGB color space
    };
#pragma pack(pop)

    FileHeader mFileHeader;
    ImageHeader mImageHeader;
    ColorHeader mColorHeader;
  };
};
