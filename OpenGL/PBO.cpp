#include <gl/glew.h>

#include "PBO.h"
#include "Common\Logger.h"

namespace gl
{

PBO::PBO()
  : mId( 0 )
{
  glGenBuffers( 1, &mId );
}

PBO::~PBO()
{
  glDeleteBuffers( 1, &mId );
}

void PBO::Allocate( const uint32_t byteCount )
{
  logger::Logger::Instance() << std::string( "PBO created with " ) << byteCount << " bytes\n";
  glBufferData( GL_PIXEL_UNPACK_BUFFER, byteCount, NULL, GL_DYNAMIC_COPY ); // last param always can be this one ?
}

void PBO::Bind()
{
  glBindBuffer( GL_PIXEL_UNPACK_BUFFER, mId );
}

void PBO::Unbind()
{
  glBindBuffer( GL_PIXEL_UNPACK_BUFFER, 0 );
}

const uint32_t& PBO::Id() const
{
  return mId;
}

rt::Color* PBO::MapPboBuffer()
{
  return reinterpret_cast<rt::Color*>( glMapBuffer( GL_PIXEL_UNPACK_BUFFER, GL_WRITE_ONLY ) );
}

void PBO::UnMapPboBuffer()
{
  glUnmapBuffer( GL_PIXEL_UNPACK_BUFFER );
}

}
