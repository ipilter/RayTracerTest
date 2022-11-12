#include "GLCanvas.h"

GLCanvas::GLCanvas( const math::uvec2& imageSize
                , wxWindow* parent
                , wxWindowID id
                , const int* attribList
                , const wxPoint& pos
                , const wxSize& size
                , long style
                , const wxString& name
                , const wxPalette& palette )
  : wxGLCanvas( parent, id, attribList, pos, size, style, name, palette )
  , mImageSize( imageSize )
{
  // OpenGL
  wxGLContextAttrs contextAttrs;
  contextAttrs.CoreProfile().OGLVersion( 4, 5 ).Robust().ResetIsolation().EndList();
  mContext = std::make_unique<wxGLContext>( this, nullptr, &contextAttrs );
  SetCurrent( *mContext );

  glewExperimental = false;
  GLenum err = glewInit();
  if ( err != GLEW_OK )
  {
    const GLubyte* msg = glewGetErrorString( err );
    throw std::exception( reinterpret_cast<const char*>( msg ) );
  }

  auto fp64 = glewGetExtension( "GL_ARB_gpu_shader_fp64" );
  fp64 = 0;
}

const math::uvec2& GLCanvas::ImageSize() const
{
  return mImageSize;
}
