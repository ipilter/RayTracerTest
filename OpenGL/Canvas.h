#pragma once

#include <map>
#include <vector>
#include <gl/glew.h> // before any gl related include
#include "../WxMain.h"
#include <wx/glcanvas.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

#include "../Math.h"
#include "../Logger.h"

#include "../OpenGL/PixelBufferObject.h"
#include "../OpenGL/Texture.h"
#include "../OpenGL/Shader.h"
#include "../OpenGL/Mesh.h"
#include "../OpenGL/Camera.h"

namespace gl
{

class Canvas : public wxGLCanvas
{
public:
  Canvas( const math::uvec2& textureSize
            , wxWindow* parent
            , wxWindowID id = wxID_ANY
            , const int* attribList = 0
            , const wxPoint& pos = wxDefaultPosition
            , const wxSize& size = wxDefaultSize
            , long style = 0L
            , const wxString& name = L"Canvas"
            , const wxPalette& palette = wxNullPalette );

  virtual ~Canvas();

  Canvas( const Canvas& ) = delete;
  Canvas( Canvas&& ) = delete;
  Canvas& operator = ( const Canvas& ) = delete;
  Canvas& operator = ( Canvas&& ) = delete;

  void UpdateTexture();
  gl::PixelBufferObject::sptr& GetRenderTarget();
  const math::uvec2& GetTextureSize() const;

private:
  void Initialize();

  void CreateMeshes();
  void CreateTextures();
  void CreateShaders();

  void OnPaint( wxPaintEvent& event );
  void OnSize( wxSizeEvent& event );
  void OnMouseMove( wxMouseEvent& event );
  void OnMouseRightDown( wxMouseEvent& event );
  void OnMouseRightUp( wxMouseEvent& event );
  void OnMouseLeftDown( wxMouseEvent& event );
  void OnMouseLeftUp( wxMouseEvent& event );
  void OnMouseLeave( wxMouseEvent& event );
  void OnMouseWheel( wxMouseEvent& event );

  math::vec2 ScreenToWorld( const math::vec2& screenSpacePoint );
  math::ivec2 WorldToImage( const math::vec2& worldSpacePoint );

  // opengl context
  std::unique_ptr<wxGLContext> mContext;

  // parameters
  math::uvec2 mTextureSize; // pixels
  math::vec2 mQuadSize; // world

  std::vector<gl::Texture::sptr> mTextures;
  std::vector<gl::PixelBufferObject::sptr> mPBOs;
  std::vector<gl::Shader::sptr> mShaders;
  std::vector<gl::Mesh::sptr> mMeshes;
  std::vector<gl::Camera::sptr> mCameras;

  // control
  bool mPanningActive;
  math::vec2 mPreviousMousePosition;
};

}
