#include <sstream>
#include <gl/glew.h>

#include "../Util.h"
#include "Shader.h"

namespace gl
{

Shader::Shader( const std::string& vertexShaderSrc, const std::string& fragentShaderSrc )
  : mVertexShader( 0 )
  , mFragmentxShader( 0 )
  , mShaderProgram( 0 )
{
  CreateShaderProgram( vertexShaderSrc, fragentShaderSrc );
}

Shader::~Shader()
{
  glDeleteShader( mVertexShader );
  glDeleteShader( mFragmentxShader );
  glDeleteProgram( mShaderProgram );
}

void Shader::Bind()
{
  glUseProgram( mShaderProgram );
}

void Shader::UnBind()
{
  glUseProgram( 0 );
}

void Shader::CreateShaderProgram( const std::string& vertexShaderSrc, const std::string& fragentShaderSrc )
{
  mShaderProgram = glCreateProgram();
  if ( mShaderProgram == 0 )
  {
    throw std::runtime_error( "cannot create shader program" );
  }

  mVertexShader = CreateShader( GL_VERTEX_SHADER, vertexShaderSrc );
  mFragmentxShader = CreateShader( GL_FRAGMENT_SHADER, fragentShaderSrc );

  glLinkProgram( mShaderProgram );
  GLint linked( GL_FALSE );
  glGetProgramiv( mShaderProgram, GL_LINK_STATUS, &linked );
  if ( linked == GL_FALSE )
  {
    int info_size( 0 );
    glGetProgramiv( mShaderProgram, GL_INFO_LOG_LENGTH, &info_size );
    std::string msg;
    if ( info_size > 0 )
    {
      std::string buffer( info_size++, ' ' );
      glGetProgramInfoLog( mShaderProgram, info_size, NULL, &buffer[0] );
      msg.swap( buffer );
    }

    std::stringstream ss;
    ss << "cannot link shader program: " << msg;
    throw std::runtime_error( ss.str() );
  }
}

uint32_t Shader::CreateShader( uint32_t kind, const std::string& src )
{
  uint32_t shaderId = glCreateShader( kind );
  if ( shaderId == 0 )
  {
    std::stringstream ss;
    ss << "cannot create shader " << kind;
    throw std::runtime_error( ss.str() );
  }

  glAttachShader( mShaderProgram, shaderId );
  const char* str[] = { src.c_str() };
  glShaderSource( shaderId, 1, str, 0 );

  glCompileShader( shaderId );

  GLint compiled( GL_FALSE );
  glGetShaderiv( shaderId, GL_COMPILE_STATUS, &compiled );
  if ( compiled == GL_FALSE )
  {
    int info_size( 0 );

    glGetShaderiv( shaderId, GL_INFO_LOG_LENGTH, &info_size );
    std::string msg;
    if ( info_size > 0 )
    {
      std::string buffer( info_size++, ' ' );
      glGetShaderInfoLog( shaderId, info_size, NULL, &buffer[0] );
      msg.swap( buffer );
    }
    std::stringstream ss;
    ss << "cannot compile shader " << kind << ". msg : " << msg;
    throw std::runtime_error( ss.str() );
  }

  return shaderId;
}

void Shader::UpdateUniform( const std::string& name, const math::mat4& matrix )
{
  GLint uniformLoc( glGetUniformLocation( mShaderProgram, name.c_str() ) );
  glUniformMatrix4fv( uniformLoc, 1, GL_FALSE, &matrix[0][0] );
}

}
