#include <sstream>

#include "GLCanvas.h"
#include "MainFrame.h"
#include "ResourceHandler.h"

#include "Common\Logger.h"
#include "Common\HostUtils.h"
#include "Common\Color.h"
#include "Common\Timer.h"

// Wrapper around MapCudaResource / UnMapCudaResource CUDA calls for safety
GLCanvas::CudaResourceGuard::CudaResourceGuard( GLCanvas& glCanvas )
  : mGLCanvas( glCanvas )
{
  mGLCanvas.MapCudaResource( mGLCanvas.mPBOs.back() );
}

rt::Color* GLCanvas::CudaResourceGuard::GetDevicePtr()
{
  return mGLCanvas.GetMappedCudaPointer( mGLCanvas.mPBOs.back() );
}

GLCanvas::CudaResourceGuard::~CudaResourceGuard()
{
  mGLCanvas.UnMapCudaResource( mGLCanvas.mPBOs.back() );
}

// OpenGL render surface with CUDA connection
GLCanvas::GLCanvas( const math::uvec2& imageSize
                    , MainFrame* mainFrame
                    , wxWindow* parent )
  : wxGLCanvas( parent )
  , mMainFrame( mainFrame )
  , mImageSize( imageSize )
  , mQuadSize( 1.0f * ( mImageSize.x / static_cast<float>( mImageSize.y ) ), 1.0f )
{
  try
  {
    if ( mMainFrame == nullptr )
    {
      throw std::runtime_error( "mainframe pointer is nullptr" );
    }

    Timer t;
    t.start();
    Initialize();
    CreateMeshes();
    CreateTextures();
    CreateShaders();
    CreateView();
    t.stop();
    logger::Logger::Instance() << "Initialize: " << t.ms() << " ms\n";
  }
  catch ( const std::exception& e )
  {
    logger::Logger::Instance() << "Cannot create Canvas: " << e.what() << "\n";
  }
}

GLCanvas::~GLCanvas()
{
  SetCurrent( *mContext );
  for ( const auto& pbo : mPBOs )
  {
    UnRegisterCudaResource( pbo );
  }
}

void GLCanvas::Resize( const math::uvec2& imageSize )
{
  Timer t;
  t.start();

  // TODO: Validate if cleanup is proper
  mImageSize = imageSize;
  mQuadSize = math::vec2( 1.0f * ( mImageSize.x / static_cast<float>( mImageSize.y ) ), 1.0f );

  mMeshes.clear();
  CreateMeshes();

  for ( const auto& pbo : mPBOs )
  {
    UnRegisterCudaResource( pbo );
  }
  mPboCudaResourceTable.clear();

  mTextures.clear();
  mPBOs.clear();

  CreateTextures();

  t.stop();
  logger::Logger::Instance() << "Resize: " << t.ms() << " ms\n";
}

const math::uvec2& GLCanvas::ImageSize() const
{
  return mImageSize;
}

void GLCanvas::UpdatePBO( rt::ColorPtr deviceImageBuffer, std::size_t deviceImageBufferSize )
{
  // TODO: do as fast as possible
  Timer t;
  t.start();

  auto pboCudaResource = GetPboCudaResource();
  cudaError_t err = cudaGraphicsMapResources( 1, &pboCudaResource );
  if ( err != cudaSuccess )
  {
    logger::Logger::Instance() << "cudaGraphicsMapResources failed. Reason: " << cudaGetErrorString( err ) << "\n";
  }
  
  rt::Color* pixelBufferPtr = nullptr;
  size_t pboSize = 0;
  err = cudaGraphicsResourceGetMappedPointer( reinterpret_cast<void**>( &pixelBufferPtr )
                                              , &pboSize
                                              , pboCudaResource );
  if ( err != cudaSuccess )
  {
    logger::Logger::Instance() << "cudaGraphicsResourceGetMappedPointer failed. Reason: " << cudaGetErrorString( err ) << "\n";
  }
  
  const bool ok = deviceImageBuffer != nullptr && pboSize == deviceImageBufferSize;
  if ( ok )
  {
    // TODO use common cuda data handler methods, like rt::CopyDeviceDataToHost
    err = cudaMemcpy( pixelBufferPtr, deviceImageBuffer, deviceImageBufferSize, cudaMemcpyDeviceToDevice);
    if ( err != cudaSuccess )
    {
      logger::Logger::Instance() << "cudaMemcpy failed. Reason: " << cudaGetErrorString( err ) << "\n";
    }
  }
  else
  {
    logger::Logger::Instance() << "device image data and PBO size mismatch\n";
  }
  
  err = cudaGraphicsUnmapResources( 1, &pboCudaResource );
  if ( err != cudaSuccess )
  {
    logger::Logger::Instance() << "cudaGraphicsUnmapResources failed. Reason: " << cudaGetErrorString( err ) << "\n";
  }

  t.stop();
  logger::Logger::Instance() << "UpdatePBO took: " << t.ms() << " ms\n";
}

void GLCanvas::UpdateTextureAndRefresh()
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
    logger::Logger::Instance() << "Error in GLCanvas::Update: " << e.what() << "\n";
  }
  catch ( ... )
  {
    logger::Logger::Instance() << "Unknown error during GLCanvas::Update\n";
  }
}

cudaGraphicsResource_t GLCanvas::GetPboCudaResource() const
{
  auto it = mPboCudaResourceTable.find( mPBOs.back()->Id() );
  if ( it == mPboCudaResourceTable.end() )
  {
    std::stringstream ss;
    ss << "GetPboCudeResource failed. PBO id is " << mPBOs.back()->Id();
    throw std::runtime_error( ss.str() );
  }
  return it->second;
}

math::vec2 GLCanvas::ScreenToWorld( const math::vec2& screen )
{
  const math::vec4 ndc( screen.x / static_cast<float>( GetSize().GetX() ) * 2.0f - 1.0f
                        , -screen.y / static_cast<float>( GetSize().GetY() ) * 2.0f + 1.0f
                        , 0.0f
                        , 1.0f );

  const math::mat4 invVpMatrix( glm::inverse( mCameras.back()->ViewProj() ) );
  const math::vec4 worldSpacePoint( invVpMatrix * ndc ); // !! never forget !!
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

gl::Camera::uptr& GLCanvas::Camera()
{
  return mCameras.back();
}

void GLCanvas::Initialize()
{
  // Event handlers
  // TODO like in MainFrame
  Bind( wxEVT_SIZE, &GLCanvas::OnSize, this );
  Bind( wxEVT_PAINT, &GLCanvas::OnPaint, this );
  Bind( wxEVT_RIGHT_DOWN, &GLCanvas::OnMouseRightDown, this );
  Bind( wxEVT_RIGHT_UP, &GLCanvas::OnMouseRightUp, this );
  Bind( wxEVT_LEFT_DOWN, &GLCanvas::OnMouseLeftDown, this );
  Bind( wxEVT_LEFT_UP, &GLCanvas::OnMouseLeftUp, this );
  Bind( wxEVT_MIDDLE_DOWN, &GLCanvas::OnMouseMiddleDown, this );
  Bind( wxEVT_MIDDLE_UP, &GLCanvas::OnMouseMiddleUp, this );
  Bind( wxEVT_MOTION, &GLCanvas::OnMouseMove, this );
  Bind( wxEVT_LEAVE_WINDOW, &GLCanvas::OnMouseLeave, this );
  Bind( wxEVT_MOUSEWHEEL, &GLCanvas::OnMouseWheel, this );

  // OpenGL
  {
    wxGLContextAttrs contextAttrs;
    contextAttrs.CoreProfile().OGLVersion( 4, 5 ).Robust().ResetIsolation().EndList();
    mContext = std::make_unique<wxGLContext>( this, nullptr, &contextAttrs );
    SetCurrent( *mContext );

    glewExperimental = false;
    GLenum err = glewInit();
    if ( err != GLEW_OK )
    {
      const GLubyte* msg = glewGetErrorString( err );
      throw std::runtime_error( reinterpret_cast<const char*>( msg ) );
    }

    // TODO: what is the safest way to ensure this (if pixel data is rgba)?
    glPixelStorei( GL_UNPACK_ALIGNMENT, 4 );
  }

  // Cuda
  {
    int gpuCount = 0;
    cudaError_t err = cudaGetDeviceCount( &gpuCount );
    if ( err != cudaSuccess || gpuCount < 1 )
    {
      throw std::runtime_error( std::string( "cudaGetDeviceCount failed: " ) + cudaGetErrorString( err ) );
    }

    cudaDeviceProp prop = { 0 };
    const int gpuId = 0;
    cudaSetDevice( gpuId );

    err = cudaGetDeviceProperties( &prop, gpuId );
    if ( err != cudaSuccess )
    {
      throw std::runtime_error( std::string( "cudaGetDeviceProperties failed: " ) + cudaGetErrorString( err ) );
    }

    err = cudaGLSetGLDevice( gpuId );
    if ( err != cudaSuccess )
    {
      throw std::runtime_error( std::string( "cudaGLSetGLDevice failed: " ) + cudaGetErrorString( err ) );
    }
  }
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
  // We are using a single pbo here as this will be our intermediate pixel storage between the GL texture and the Tracer's own pixel data
  // The rendered texture will be updated from this buffer when a render phase is done
  mPBOs.push_back( std::make_unique<gl::PBO>() );
  mPBOs.back()->Bind();
  // allocate PBO memory
  mPBOs.back()->Allocate( byteCount );
  
  // register as cuda interop resource
  RegisterCudaResource( mPBOs.back() );
  
  // init PBO data
  {
    rt::Color* devicePixelBufferPtr( mPBOs.back()->MapPboBuffer() );
    std::fill( devicePixelBufferPtr, devicePixelBufferPtr + pixelCount, rt::GetColor( 10, 10, 10 ) );
    mPBOs.back()->UnMapPboBuffer();
  }

  // create texture from PBO pixels
  mTextures.back()->Bind();
  mTextures.back()->CreateFromPBO();
  mTextures.back()->Unbind();
  mPBOs.back()->Unbind();

  logger::Logger::Instance() << "Texture with dimensions " << mTextures.back()->Size() << " created\n";
}

void GLCanvas::CreateShaders()
{
  const std::string vertexShaderSrc( resource::LoadString( VERTEX_SHADER, TEXTFILE ) );
  const std::string fragentShaderSrc( resource::LoadString( FRAGMENT_SHADER, TEXTFILE ) );
  mShaders.push_back( std::make_unique<gl::Shader>( vertexShaderSrc, fragentShaderSrc ) );
}

void GLCanvas::CreateView()
{
  mCameras.push_back( std::make_unique<gl::Camera>( math::vec3( -mQuadSize.x / 2.0f, -mQuadSize.y / 2.0f, 0.0f ) ) );
}

void GLCanvas::RegisterCudaResource( const gl::PBO::uptr& pbo )
{
  auto it = mPboCudaResourceTable.insert( std::make_pair( pbo->Id(), cudaGraphicsResource_t( 0 ) ) );
  if ( it.second )
  {
    cudaError_t err = cudaGraphicsGLRegisterBuffer( &it.first->second, pbo->Id(), cudaGraphicsMapFlagsWriteDiscard );
    if ( err != cudaSuccess )
    {
      throw std::runtime_error( std::string( "cudaGraphicsGLRegisterBuffer failed: " ) + cudaGetErrorString( err ) );
    }
  }
}

void GLCanvas::UnRegisterCudaResource( const gl::PBO::uptr& pbo )
{
  cudaError_t err = cudaGraphicsUnregisterResource( mPboCudaResourceTable[pbo->Id()] );
  if ( err != cudaSuccess )
  {
    throw std::runtime_error( std::string( "cudaGraphicsUnregisterResource failed: " ) + cudaGetErrorString( err ) );
  }
}

void GLCanvas::MapCudaResource( const gl::PBO::uptr& pbo )
{
  cudaError_t err = cudaGraphicsMapResources( 1, &mPboCudaResourceTable[pbo->Id()] ); // TODO searching every time
  if ( err != cudaSuccess )
  {
    throw std::runtime_error( std::string( "cudaGraphicsMapResources failed: " ) + cudaGetErrorString( err ) );
  }
}

void GLCanvas::UnMapCudaResource( const gl::PBO::uptr& pbo )
{
  cudaError_t err = cudaGraphicsUnmapResources( 1, &mPboCudaResourceTable[pbo->Id()] ); // TODO searching every time
  if ( err != cudaSuccess )
  {
    throw std::runtime_error( std::string( "cudaGraphicsUnmapResources failed: " ) + cudaGetErrorString( err ) );
  }
}

rt::Color* GLCanvas::GetMappedCudaPointer( const gl::PBO::uptr& pbo )
{
  rt::Color* ptr = nullptr;
  size_t mapped_size = 0;

  // TODO searching every time
  cudaError_t err = cudaGraphicsResourceGetMappedPointer( reinterpret_cast<void**>( &ptr ), &mapped_size, mPboCudaResourceTable[pbo->Id()] );
  if ( err != cudaSuccess )
  {
    throw std::runtime_error( std::string( "cudaGraphicsResourceGetMappedPointer failed: " ) + cudaGetErrorString( err ) );
  }
  return ptr;
}

void GLCanvas::OnPaint( wxPaintEvent& /*event*/ )
{
  // reset
  SetCurrent( *mContext );

  // clear frame
  glClearColor( 0.15f, 0.15f, 0.15f, 1.0f );
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
  PropagateEventToMainFrame( event );
}

void GLCanvas::OnMouseWheel( wxMouseEvent& event )
{
  PropagateEventToMainFrame( event );
}

void GLCanvas::OnMouseRightDown( wxMouseEvent& event )
{
  PropagateEventToMainFrame( event );
}

void GLCanvas::OnMouseRightUp( wxMouseEvent& event )
{
  PropagateEventToMainFrame( event );
}

void GLCanvas::OnMouseLeftDown( wxMouseEvent& event )
{
  PropagateEventToMainFrame( event );
}

void GLCanvas::OnMouseLeftUp( wxMouseEvent& event )
{
  PropagateEventToMainFrame( event );
}

void GLCanvas::OnMouseMiddleDown( wxMouseEvent& event )
{
  PropagateEventToMainFrame( event );
}

void GLCanvas::OnMouseMiddleUp( wxMouseEvent& event )
{
  PropagateEventToMainFrame( event );
}

void GLCanvas::OnMouseLeave( wxMouseEvent& event )
{
  PropagateEventToMainFrame( event );
}

void GLCanvas::PropagateEventToMainFrame( wxEvent& event )
{
  mMainFrame->GetEventHandler()->ProcessEvent( event );
}
