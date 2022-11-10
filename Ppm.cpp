#include "ppm.h"

namespace rt
{

Ppm::Pixel::Pixel( const Ppm::Pixel::ChannelType r, const Ppm::Pixel::ChannelType g, const Ppm::Pixel::ChannelType b )
  : mRed( r ), mGreen( g ), mBlue( b )
{ }

Ppm::Pixel::ChannelType Ppm::Pixel::maxColor()
{
  return 255;
}

Ppm::Pixel& Ppm::Pixel::operator += ( const Ppm::Pixel& rhs )
{
  mRed += rhs.mRed;
  mGreen += rhs.mGreen;
  mBlue += rhs.mBlue;
  return *this;
}

Ppm::Pixel Ppm::Pixel::operator + ( const Ppm::Pixel& rhs )
{
  return Ppm::Pixel( mRed + rhs.mRed
                     , mGreen + rhs.mGreen
                     , mBlue + rhs.mBlue );
}

Ppm::Pixel& Ppm::Pixel::operator -= ( const Ppm::Pixel& rhs )
{
  mRed -= rhs.mRed;
  mGreen -= rhs.mGreen;
  mBlue -= rhs.mBlue;
  return *this;
}

Ppm::Pixel Ppm::Pixel::operator - ( const Ppm::Pixel& rhs )
{
  return Pixel( mRed - rhs.mRed
                , mGreen - rhs.mGreen
                , mBlue - rhs.mBlue );
}

Ppm::Pixel Ppm::Pixel::operator * ( const float s ) const
{
  return Pixel( static_cast<Ppm::Pixel::ChannelType>( mRed * s )
                , static_cast<Ppm::Pixel::ChannelType>( mGreen * s )
                , static_cast<Ppm::Pixel::ChannelType>( mBlue * s ) );
}

Ppm::Ppm( const math::uvec2& size, const Ppm::Pixel& color )
  : mSize( size )
  , mImageData( size.x * size.y, color )
{ }

Ppm::sptr Ppm::create( const math::uvec2& size, const Pixel& color )
{
  return std::make_shared<Ppm>( size, color );
}

void Ppm::setPixel( const math::uvec2& pos, const Ppm::Pixel& p )
{
  mImageData[pos.x + pos.y * mSize.x] = p;
}

Ppm::Pixel Ppm::getPixel( const math::uvec2& pos ) const
{
  return mImageData[pos.x + pos.y * mSize.x];
}

void Ppm::fill( const Ppm::Pixel& color )
{
  std::fill( mImageData.begin(), mImageData.end(), color );
}

const math::uvec2& Ppm::size() const
{
  return mSize;
}

uint64_t Ppm::bytes() const
{
  return mImageData.size() * sizeof( Pixel );
}

Ppm::Pixel* Ppm::memory()
{
  return &mImageData[0];
}

void Ppm::save( const std::string& file ) const
{
  std::ofstream stream( file.c_str(), std::ios::binary | std::ios::out );
  if ( !stream.is_open() )
  {
    throw std::runtime_error( std::string( "could not open ppm file: " ) + file );
  }
  stream << "P6" << std::endl << mSize.x << " " << mSize.y << std::endl << (int) Pixel::maxColor() << std::endl;
  stream.write( reinterpret_cast<const char*>( &mImageData[0] ), bytes() );
}

}
