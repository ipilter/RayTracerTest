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
  ~GLCanvas();

  void UpdateTexture();
 
  const math::uvec2& ImageSize() const;

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

private:
  std::unique_ptr<wxGLContext> mContext;

  math::uvec2 mImageSize; // pixels
  math::vec2 mQuadSize;   // world

  std::vector<gl::Texture::uptr> mTextures;
  std::vector<gl::PBO::uptr> mPBOs;
  std::vector<gl::Shader::uptr> mShaders;
  std::vector<gl::Mesh::uptr> mMeshes;
  std::vector<gl::Camera::uptr> mCameras;

  bool mPanningActive;
  math::vec2 mPreviousMousePosition;

private:
  GLCanvas( const GLCanvas& ) = delete;
  GLCanvas( GLCanvas&& ) = delete;
  GLCanvas& operator = ( const GLCanvas& ) = delete;
  GLCanvas& operator = ( GLCanvas&& ) = delete;
};
