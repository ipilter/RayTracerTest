#pragma once

#include <map>
#include <gl/glew.h> // before any gl related include

#include "WxMain.h"
#include <wx/glcanvas.h>
#include <cuda_gl_interop.h>

#include "Common\Math.h"
#include "Common\Sptr.h"

#include "OpenGL\Camera.h"
#include "OpenGL\Mesh.h"
#include "OpenGL\PBO.h"
#include "OpenGL\Shader.h"
#include "OpenGL\Texture.h"

class MainFrame;

class GLCanvas : public wxGLCanvas, public virtual ISptr<GLCanvas>
{
public:
  class CudaResourceGuard
  {
  public:
    CudaResourceGuard( GLCanvas& glCanvas );
    ~CudaResourceGuard();

    rt::color_t* GetDevicePtr();

  private:
    GLCanvas& mGLCanvas;
  };

public:
  GLCanvas( const math::uvec2& imageSize
            , MainFrame* mainFrame
            , wxWindow* parent );
  ~GLCanvas();

  void Resize( const math::uvec2& imageSize );
  const math::uvec2& ImageSize() const;

  void UpdateTextureAndRefresh();

  cudaGraphicsResource_t GetPboCudeResource() const;

private:
  void Initialize();
  void CreateMeshes();
  void CreateTextures();
  void CreateShaders();
  void CreateView();

  math::vec2 ScreenToWorld( const math::vec2& screenSpacePoint );
  math::ivec2 WorldToImage( const math::vec2& worldSpacePoint );

  void RegisterCudaResource( const gl::PBO::uptr& pbo );
  void UnRegisterCudaResource( const gl::PBO::uptr& pbo );

  void MapCudaResource( const gl::PBO::uptr& pbo );
  void UnMapCudaResource( const gl::PBO::uptr& pbo );
  rt::color_t* GetMappedCudaPointer( const gl::PBO::uptr& pbo );

  void OnPaint( wxPaintEvent& event );
  void OnSize( wxSizeEvent& event );
  void OnMouseMove( wxMouseEvent& event );
  void OnMouseRightDown( wxMouseEvent& event );
  void OnMouseRightUp( wxMouseEvent& event );
  void OnMouseLeftDown( wxMouseEvent& event );
  void OnMouseLeftUp( wxMouseEvent& event );
  void OnMouseMiddleDown( wxMouseEvent& event );
  void OnMouseMiddleUp( wxMouseEvent& event );
  void OnMouseLeave( wxMouseEvent& event );
  void OnMouseWheel( wxMouseEvent& event );

  void PropagateEventToMainFrame( wxEvent& event );

  std::unique_ptr<wxGLContext> mContext;

  MainFrame* mMainFrame;
  math::uvec2 mImageSize; // pixels
  math::vec2 mQuadSize;   // world

  std::vector<gl::Texture::uptr> mTextures;
  std::vector<gl::PBO::uptr> mPBOs;
  std::vector<gl::Shader::uptr> mShaders;
  std::vector<gl::Mesh::uptr> mMeshes;
  std::vector<gl::Camera::uptr> mCameras;

  std::map<uint32_t, cudaGraphicsResource_t> mPboCudaResourceTable;

  bool mPanningActive;
  math::vec2 mPreviousMouseScreenPosition;

  GLCanvas( const GLCanvas& ) = delete;
  GLCanvas( GLCanvas&& ) = delete;
  GLCanvas& operator = ( const GLCanvas& ) = delete;
  GLCanvas& operator = ( GLCanvas&& ) = delete;
};
