#include <gl/glew.h>

#include "PBO.h"

PBO::PBO()
{
  glGenBuffers( 1, &mId );
}

PBO::~PBO()
{
  glDeleteBuffers( 1, &mId );
}

void PBO::Allocate( uint32_t byteCount )
{
  glBufferData( GL_PIXEL_UNPACK_BUFFER, byteCount, NULL, GL_DYNAMIC_COPY ); // last param always can be this one ?
}

uint32_t PBO::Id() const
{
  return mId;
}

void PBO::Bind()
{
  glBindBuffer( GL_PIXEL_UNPACK_BUFFER, mId );
}

void PBO::Unbind()
{
  glBindBuffer( GL_PIXEL_UNPACK_BUFFER, 0 );
}

uint32_t* PBO::MapBuffer()
{
  return reinterpret_cast<uint32_t*>( glMapBuffer( GL_PIXEL_UNPACK_BUFFER, GL_WRITE_ONLY ) );
}

void PBO::UnmapBuffer()
{
  glUnmapBuffer( GL_PIXEL_UNPACK_BUFFER );
}
