#pragma once

#include <gl/glew.h> // before any gl related include
#include "WxMain.h"
#include <wx/glcanvas.h>

#include "Common\Math.h"
#include "Common\Sptr.h"

#include "OpenGL\Camera.h"
#include "OpenGL\Mesh.h"
#include "OpenGL\PBO.h"
#include "OpenGL\Shader.h"
#include "OpenGL\Texture.h"

class GLCanvas : public wxGLCanvas, public virtual ISptr<GLCanvas>
{
public:
  GLCanvas( const math::uvec2& imageSize
          , wxWindow* parent
          , wxWindowID id = wxID_ANY
          , const int* attribList = 0
          , const wxPoint& pos = wxDefaultPosition
          , const wxSize& size = wxDefaultSize
          , long style = 0L
          , const wxString& name = L"GLCanvas"
          , const wxPalette& palette = wxNullPalette );

  const math::uvec2& ImageSize() const;

private:
  // opengl context
  std::unique_ptr<wxGLContext> mContext;

  math::uvec2 mImageSize;

  gl::Camera::uptr mCamera;
  gl::Mesh::uptr mMesh;
  gl::PBO::uptr mPBO;
  gl::Shader::uptr mShader;
  gl::Texture::uptr mTexture;
};
