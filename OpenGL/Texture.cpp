#include <gl/glew.h>

#include "../Logger.h"
#include "Texture.h"

namespace gl
{

Texture::Texture( const uint32_t width, const uint32_t height, const uint32_t wrap )
  : mWidth( width )
  , mHeight( height )
  , mChannelCount( 4 ) // GL_RGBA
{
  glGenTextures( 1, &mId );
  glBindTexture( GL_TEXTURE_2D, mId );
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, wrap );
  glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, wrap );
  glBindTexture( GL_TEXTURE_2D, 0 );
}

Texture::~Texture()
{
  glDeleteTextures( 1, &mId );
}

void Texture::Bind()
{
  glBindTexture( GL_TEXTURE_2D, mId );
}

void Texture::Unbind()
{
  glBindTexture( GL_TEXTURE_2D, 0 );
}

void Texture::BindTextureUnit( const uint32_t unitId )
{
  glBindTextureUnit( unitId, mId );
}

void Texture::UnbindTextureUnit()
{
  glBindTextureUnit( 0, 0 );
}

void Texture::CreateFromArray( const uint32_t* array )  // TODO: let channel count and bytes per channel be also param from outside (fully descirbe the incoming pixel data)
{
  glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA8, mWidth, mHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, array );
}

void Texture::CreateFromPBO()  // TODO: let channel count and bytes per channel be also param from outside (fully descirbe the incoming pixel data)
{
  glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA8, mWidth, mHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0 );
}

void Texture::UpdateFromPBO()
{
  glPixelStorei( GL_UNPACK_ROW_LENGTH, 0 );
  glTexSubImage2D( GL_TEXTURE_2D, 0, 0, 0, mWidth, mHeight, GL_RGBA, GL_UNSIGNED_BYTE, 0 );
}

void Texture::UpdateFromPBO( uint32_t regionPosX, uint32_t regionPosY, uint32_t regionWidth, uint32_t regionHeight )
{
  glPixelStorei( GL_UNPACK_ROW_LENGTH, mWidth );
  glTexSubImage2D( GL_TEXTURE_2D, 0
                   , regionPosX, regionPosY
                   , regionWidth, regionHeight
                   , GL_RGBA, GL_UNSIGNED_BYTE
                   , reinterpret_cast<void*>( regionPosX * 4ull + regionPosY * 4ull * mWidth ) );
  glPixelStorei( GL_UNPACK_ROW_LENGTH, 0 );
}

uint32_t Texture::Width() const
{
  return mWidth;
}

uint32_t Texture::Height() const
{
  return mHeight;
}

uint32_t Texture::ChannelCount() const
{
  return mChannelCount;
}

uint32_t Texture::BytesPerChannel() const
{
  const auto pixelType = GL_UNSIGNED_BYTE;

  switch ( pixelType )
  {
    case GL_UNSIGNED_BYTE:
      return sizeof( unsigned char );
    default:
    {
      logger::Logger::Instance() << "unknown pixel type in Texture::BytesPerChannel: " << pixelType;
    }
  }
}

}
