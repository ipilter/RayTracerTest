#include <sstream>

#include "GLCanvas.h"
#include "Common\Logger.h"
#include "Common\HostUtils.h"

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
  , mQuadSize( 1.0 * ( mImageSize.x / static_cast<float>( mImageSize.y ) ), 1.0 )
  , mPanningActive( false )
  , mPreviousMousePosition( 0.0f, 0.0f )
{
  try
  {
    Initialize();

    CreateMeshes();
    CreateShaders();
    CreateTextures();

    mCameras.push_back( std::make_unique<gl::Camera>( math::vec3( -mQuadSize.x / 2.0, -mQuadSize.y / 2.0, 0.0 ) ) );
  }
  catch ( const std::exception& e )
  {
    logger::Logger::Instance() << "Cannot create Canvas: " << e.what() << "\n";
  }
}

GLCanvas::~GLCanvas()
{
  SetCurrent( *mContext );
}

const math::uvec2& GLCanvas::ImageSize() const
{
  return mImageSize;
}
void GLCanvas::UpdateTexture()
{
  try
  {
    mPBOs.back()->Bind();

    mTextures.back()->Bind();
    mTextures.back()->UpdateFromPBO();
    mTextures.back()->Unbind();

    mPBOs.back()->Unbind();
    Refresh();
  }
  catch ( const std::exception& e )
  {
    std::stringstream ss;
    ss << "Step error: " << e.what();
  }
  catch ( ... )
  {
    std::stringstream ss;
    ss << "unknown Step error: ";
  }
}

void GLCanvas::Initialize()
{
  // Event handlers
  Bind( wxEVT_SIZE, &GLCanvas::OnSize, this );
  Bind( wxEVT_PAINT, &GLCanvas::OnPaint, this );
  Bind( wxEVT_RIGHT_DOWN, &GLCanvas::OnMouseRightDown, this );
  Bind( wxEVT_RIGHT_UP, &GLCanvas::OnMouseRightUp, this );
  Bind( wxEVT_LEFT_DOWN, &GLCanvas::OnMouseLeftDown, this );
  Bind( wxEVT_LEFT_UP, &GLCanvas::OnMouseLeftUp, this );
  Bind( wxEVT_MOTION, &GLCanvas::OnMouseMove, this );
  Bind( wxEVT_LEAVE_WINDOW, &GLCanvas::OnMouseLeave, this );
  Bind( wxEVT_MOUSEWHEEL, &GLCanvas::OnMouseWheel, this );

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
  logger::Logger::Instance() << "GL_ARB_gpu_shader_fp64 " << ( fp64 == 1 ? "supported" : "not supported" ) << "\n";

  // TODO: what is the safest way to ensure this (if pixel data is rgba)?
  glPixelStorei( GL_UNPACK_ALIGNMENT, 4 );

  // Cuda
  //int gpuCount = 0;
  //cudaError_t error_test = cudaGetDeviceCount( &gpuCount );
  //if ( error_test != cudaSuccess )
  //{
  //  dynamic_cast<MainFrame*>( GetParent()->GetParent() )->AddLogMessage( "cudaGetDeviceCount failed" );
  //  return;
  //}

  //cudaDeviceProp prop = { 0 };
  //int gpuId = 0;
  //error_test = cudaGetDeviceProperties( &prop, gpuId );
  //if ( error_test != cudaSuccess )
  //{
  //  dynamic_cast<MainFrame*>( GetParent()->GetParent() )->AddLogMessage( "cudaGetDeviceProperties failed" );
  //  return;
  //}

  //error_test = cudaGLSetGLDevice( gpuId );
  //if ( error_test != cudaSuccess )
  //{
  //  return;
  //}
}

void GLCanvas::CreateMeshes()
{
  const std::vector<float> vertices = { 0.0f,          0.0f,        0.0f, 1.0f // vtx bl
    , mQuadSize.x, 0.0f,        1.0f, 1.0f // vtx br
    , 0.0f,        mQuadSize.y, 0.0f, 0.0f // vtx tl
    , mQuadSize.x, mQuadSize.y, 1.0f, 0.0f // vtx tr
  };

  const std::vector<uint32_t> indices = { 0, 1, 2,  1, 3, 2 };

  mMeshes.push_back( std::make_unique<gl::Mesh>( vertices, indices ) );
}

void GLCanvas::CreateTextures()
{
  // create texture
  // TODO: parametrize hardcoded params: pixel format, type, etc
  int maxTextureSize = 0;
  glGetIntegerv( GL_MAX_TEXTURE_SIZE, &maxTextureSize );

  mTextures.push_back( std::make_unique<gl::Texture>( math::uvec2( glm::min( mImageSize.x, static_cast<uint32_t>( maxTextureSize ) )
                                                                   , glm::min( mImageSize.y, static_cast<uint32_t>( maxTextureSize ) ) ) ) );

  const size_t pixelCount( mTextures.back()->Size().x * static_cast<size_t>( mTextures.back()->Size().y ) );
  const size_t byteCount( pixelCount * sizeof( uint32_t ) );

  // create PBO
  // TODO: multiple for float/triple buffering
  mPBOs.push_back( std::make_unique<gl::PBO>() );
  mPBOs.back()->Bind();
  mPBOs.back()->Allocate( byteCount );
  //mPBOs.back()->RegisterCudaResource();

  // init PBO data with zero
  //{
  //  uint32_t* devicePixelBufferPtr( mPBOs.back()->MapPboBuffer() );
  //  std::fill( devicePixelBufferPtr, devicePixelBufferPtr + pixelCount, 0 );
  //  mPBOs.back()->UnmapPboBuffer();
  //}

  // create texture from PBO pixels
  mTextures.back()->Bind();
  mTextures.back()->CreateFromPBO();
  mTextures.back()->Unbind();

  // unbind PBO
  mPBOs.back()->Unbind();

  std::stringstream ss;
  ss << "Texture with dimensions " << mTextures.back()->Size().x << "x" << mTextures.back()->Size().y << " created";
}

void GLCanvas::CreateShaders()
{
  // TODO: parametrize hardcoded shader file paths
  const std::string vertexShaderSrc( util::ReadTextFile( "shader.vert" ) );
  const std::string fragentShaderSrc( util::ReadTextFile( "shader.frag" ) );

  mShaders.push_back( std::make_unique<gl::Shader>( vertexShaderSrc, fragentShaderSrc ) );
}

math::vec2 GLCanvas::ScreenToWorld( const math::vec2& screen )
{
  const math::vec4 ndc( screen.x / static_cast<float>( GetSize().GetX() ) * 2.0 - 1.0
                        , -screen.y / static_cast<float>( GetSize().GetY() ) * 2.0 + 1.0
                        , 0.0
                        , 1.0 );

  const math::mat4 invVpMatrix( glm::inverse( mCameras.back()->ViewProj() ) );
  const math::vec4 worldSpacePoint( invVpMatrix * ndc ); // !!
  return math::vec2( worldSpacePoint );
}

math::ivec2 GLCanvas::WorldToImage( const math::vec2& worldSpacePoint )
{
  const float x( worldSpacePoint.x / mQuadSize.x * mTextures.back()->Size().x );
  const float y( -worldSpacePoint.y / mQuadSize.y * mTextures.back()->Size().y + mTextures.back()->Size().y );  // texture`s and world`s y are in the opposite order

  if ( x < 0.0f || x >= static_cast<float>( mTextures.back()->Size().x )
       || y < 0.0f || y >= static_cast<float>( mTextures.back()->Size().y ) )
  {
    return math::ivec2( -1, -1 );
  }
  return math::ivec2( glm::floor( x ), glm::floor( y ) );
}

void GLCanvas::OnPaint( wxPaintEvent& /*event*/ )
{
  // reset
  SetCurrent( *mContext );

  // clear frame
  glClearColor( 0.05f, 0.05f, 0.05f, 1.0f );
  glClear( GL_COLOR_BUFFER_BIT );

  // apply current material
  mShaders.back()->Bind();
  mTextures.back()->BindTextureUnit( 0 );

  //update shaders and draw
  mShaders.back()->UpdateUniform( "vpMatrix", mCameras.back()->ViewProj() );
  mMeshes.back()->Draw();

  // clean up
  mTextures.back()->UnbindTextureUnit();
  mShaders.back()->UnBind();

  // swap render target ( textures )
  SwapBuffers();
}

void GLCanvas::OnSize( wxSizeEvent& event )
{
  glViewport( 0, 0, event.GetSize().GetX(), event.GetSize().GetY() );

  const float aspectRatio( static_cast<float>( event.GetSize().GetX() ) / static_cast<float>( event.GetSize().GetY() ) );
  float xSpan( 1.0f );
  float ySpan( 1.0f );

  if ( aspectRatio > 1.0f )
  {
    xSpan *= aspectRatio;
  }
  else
  {
    ySpan = xSpan / aspectRatio;
  }

  mCameras.back()->Ortho( xSpan, ySpan );
}

void GLCanvas::OnMouseMove( wxMouseEvent& event )
{
  if ( mPanningActive )
  {
    const math::ivec2 screenPos( event.GetX(), event.GetY() );
    const math::vec2 worldPos( ScreenToWorld( screenPos ) );
    const math::ivec2 imagePos( WorldToImage( worldPos ) );

    const math::vec2 mouse_delta( worldPos - ScreenToWorld( mPreviousMousePosition ) );

    mCameras.back()->Translate( math::vec3( mouse_delta, 0.0f ) );

    mPreviousMousePosition = screenPos;
    Refresh();
  }
}

void GLCanvas::OnMouseWheel( wxMouseEvent& event )
{
  const float scaleFactor( 0.1f );
  const float scale( event.GetWheelRotation() < 0 ? 1.0f - scaleFactor : 1.0f + scaleFactor );

  const math::vec2 screenFocusPoint( static_cast<float>( event.GetX() ), static_cast<float>( event.GetY() ) );
  const math::vec2 worldFocusPoint( ScreenToWorld( screenFocusPoint ) );

  mCameras.back()->Translate( math::vec3( worldFocusPoint, 0.0f ) );
  mCameras.back()->Scale( math::vec3( scale, scale, 1.0f ) );
  mCameras.back()->Translate( math::vec3( -worldFocusPoint, 0.0f ) );

  Refresh();
}

void GLCanvas::OnMouseRightDown( wxMouseEvent& event )
{
  mPreviousMousePosition = math::vec2( static_cast<float>( event.GetX() ), static_cast<float>( event.GetY() ) );
  mPanningActive = true;
}

void GLCanvas::OnMouseRightUp( wxMouseEvent& /*event*/ )
{
  mPanningActive = false;
}

void GLCanvas::OnMouseLeftDown( wxMouseEvent& event )
{
  const math::ivec2 screenPos( event.GetX(), event.GetY() );
  const math::vec2 worldPos( ScreenToWorld( screenPos ) );
  const math::ivec2 imagePos( WorldToImage( worldPos ) );

  if ( imagePos == math::ivec2( -1, -1 ) )
  {
    return;
  }

  Refresh();
}

void GLCanvas::OnMouseLeftUp( wxMouseEvent& /*event*/ )
{}

void GLCanvas::OnMouseLeave( wxMouseEvent& /*event*/ )
{
  mPanningActive = false;
}
