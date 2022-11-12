#include <gl/glew.h>

#include "Mesh.h"

namespace gl
{

Mesh::Mesh( const std::vector<float>& vertices, const std::vector<uint32_t>& indices )
  : mVbo( 0 )
  , mIbo( 0 )
  , mVao( 0 )
  , mIndexCount( static_cast<uint32_t>( indices.size() ) )
{
  glGenBuffers( 1, &mVbo );
  glBindBuffer( GL_ARRAY_BUFFER, mVbo );
  glBufferData( GL_ARRAY_BUFFER, sizeof( float ) * vertices.size(), &vertices.front(), GL_STATIC_DRAW );

  glGenBuffers( 1, &mIbo );
  glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, mIbo );
  glBufferData( GL_ELEMENT_ARRAY_BUFFER, sizeof( unsigned ) * indices.size(), &indices.front(), GL_STATIC_DRAW );

  glGenVertexArrays( 1, &mVao );
  glBindVertexArray( mVao );
  glEnableVertexAttribArray( 0 );
  glEnableVertexAttribArray( 1 );
  glBindBuffer( GL_ARRAY_BUFFER, mVbo );
  glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, mIbo );

  const uint32_t stride( 4 * sizeof( float ) );
  const size_t vertexOffset( 0 );
  const size_t texelOffset( 2 * sizeof( float ) );
  glVertexAttribPointer( 0, 2, GL_FLOAT, GL_FALSE, stride, reinterpret_cast<void*>( vertexOffset ) );
  glVertexAttribPointer( 1, 2, GL_FLOAT, GL_FALSE, stride, reinterpret_cast<void*>( texelOffset ) );
  glBindVertexArray( 0 );
}

Mesh::~Mesh()
{
  glDeleteVertexArrays( 1, &mVao );
  glDeleteBuffers( 1, &mVbo );
  glDeleteBuffers( 1, &mIbo );
}

void Mesh::Draw() const
{
  glBindVertexArray( mVao );
  glDrawElements( GL_TRIANGLES, static_cast<GLsizei>( mIndexCount ), GL_UNSIGNED_INT, 0 );
  glBindVertexArray( 0 );
}

}
