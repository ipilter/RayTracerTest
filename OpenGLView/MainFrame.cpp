#include <wx\filedlg.h> // Must be before any std string include. tonns of unsage usage errors otherwise in wxWidgets code
#include <wx\filename.h>

#include <sstream>
#include <memory>
#include <functional>

#include <cudart_platform.h>

#include "MainFrame.h"
#include "RayTracer\RayTracer.h"
#include "Common\HostUtils.h"
#include "Common\Bitmap.h"
#include "Common\Logger.h"
#include "Common\Math.h"

MainFrame::MainFrame( const math::uvec2& imageSize
                      , const uint32_t sampleCount
                      , const float fov
                      , const float focalLength
                      , const float aperture
                      , wxWindow* parent
                      , std::wstring title
                      , const wxPoint& pos
                      , const wxSize& size )
  : wxFrame( parent, wxID_ANY, title, pos, size )
  , mMainSplitter( new wxSplitterWindow( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxSP_BORDER | wxSP_LIVE_UPDATE ) )
  , mLeftSplitter( new wxSplitterWindow( mMainSplitter, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxSP_BORDER | wxSP_LIVE_UPDATE ) )
  , mMainPanel( new wxPanel( mMainSplitter ) )
  , mControlPanel( new wxPanel( mMainSplitter ) )
  , mLogTextBox( new wxTextCtrl( mLeftSplitter, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxTE_MULTILINE | wxTE_READONLY ) )
  , mGLCanvas( std::make_unique<GLCanvas>( imageSize, this, mLeftSplitter ) )
  , mRayTracer( std::make_unique<rt::RayTracer>( imageSize, fov, focalLength, aperture ) ) // set some default camera parameters here
  , mCameraModeActive( false )
  , mPreviousMouseScreenPosition( 0.0f, 0.0f )
{
  // attach this object to the logger. Any log message sent to the log will propagated to the OnLogMessage callback
  logger::Logger::Instance().SetMessageCallback( std::bind( &MainFrame::OnLogMessage, this, std::placeholders::_1 ) );

  // Parameters
  mParameterControls.push_back( new NamedTextControl( mControlPanel, wxID_ANY, "Width", util::ToString( imageSize.x )
                                                                        , 100.0f, 1.0f, 1000.0f
                                                                        , 1.0f, 100000.0f ) );
  mParameterControls.push_back( new NamedTextControl( mControlPanel, wxID_ANY, "Height", util::ToString( imageSize.y )
                                                       , 100.0f, 1.0f, 1000.0f
                                                       , 1.0f, 100000.0f ) );
  mParameterControls.push_back( new NamedTextControl( mControlPanel, wxID_ANY, "Samples", util::ToString( sampleCount )
                                                        , 1.0f, 1.0f, 100.0f
                                                        , 1.0f, 10000.0f ) );
  mParameterControls.push_back( new NamedTextControl( mControlPanel, wxID_ANY, "Fov", util::ToString( fov )
                                                    , 1.0f, 0.1f, 5.0f
                                                    , 1.0f, 179.0f ) );
  mParameterControls.push_back( new NamedTextControl( mControlPanel, wxID_ANY, "Focal l.", util::ToString( focalLength )
                                                         , 10.0f, 1.0f, 50.0f
                                                         , 1.0f, 10000.0f ) );
  mParameterControls.push_back( new NamedTextControl( mControlPanel, wxID_ANY, "Aperture", util::ToString( aperture )
                                                      , 1.0f, 0.1f, 2.0f
                                                      , 0.0f, 22.0f ) ); // min == 0 = pinhole

  // Buttons
  mButtons.push_back( std::make_pair( "Render", 
                                      std::make_pair( new wxButton( mControlPanel, wxID_ANY, "Render" )
                                                      , std::bind( &MainFrame::OnRenderButton, this, std::placeholders::_1 ) ) ) );
  mButtons.push_back( std::make_pair( "Stop", 
                                      std::make_pair( new wxButton( mControlPanel, wxID_ANY, "Stop" )
                                                      , std::bind( &MainFrame::OnStopButton, this, std::placeholders::_1 ) ) ) );
  mButtons.push_back( std::make_pair( "Resize", 
                                      std::make_pair( new wxButton( mControlPanel, wxID_ANY, "Resize" )
                                                      , std::bind( &MainFrame::OnResizeButton, this, std::placeholders::_1 ) ) ) );
  mButtons.push_back( std::make_pair( "Save", 
                                      std::make_pair( new wxButton( mControlPanel, wxID_ANY, "Save" )
                                                      , std::bind( &MainFrame::OnSaveButton, this, std::placeholders::_1 ) ) ) );

  // TODO add some default values for them
  InitializeUIElements();
}

MainFrame::~MainFrame()
{
  // detach this object from the logger
  logger::Logger::Instance().SetMessageCallback();
}

void MainFrame::InitializeUIElements()
{
  try
  {
    // Place it on the screen center
    CenterOnScreen();

    wxBoxSizer* controlSizer( new wxBoxSizer( wxVERTICAL ) );
    for ( auto ctrl : mParameterControls )
    {
      controlSizer->Add( ctrl, 0, wxEXPAND );
    }

    for ( auto btn : mButtons )
    {
      controlSizer->Add( btn.second.first, 0, wxEXPAND );
      btn.second.first->Bind( wxEVT_COMMAND_BUTTON_CLICKED, btn.second.second );
    }

    mControlPanel->SetSizer( controlSizer );

    mLeftSplitter->SplitHorizontally( mGLCanvas.get(), mLogTextBox, -100 );
    mLeftSplitter->SetMinimumPaneSize( 100 );
    mLeftSplitter->SetSashGravity( 1 );

    mMainSplitter->SplitVertically( mLeftSplitter, mControlPanel, -200 );
    mMainSplitter->SetMinimumPaneSize( 200 );
    mMainSplitter->SetSashGravity( 1 );

    // Some colors
    for ( auto ctrl : mParameterControls )
    {
      ctrl->SetBackgroundColour( wxColor( 115, 115, 115 ) );
    }

    // TODO use common anchestor's ptr in a loop instead these
    mLogTextBox->SetBackgroundColour( wxColor( 065, 065, 065 ) );
    mLogTextBox->SetForegroundColour( wxColor( 200, 200, 200 ) );
    mControlPanel->SetBackgroundColour( wxColor( 075, 075, 075 ) );

    Bind( wxEVT_LEFT_DOWN, &MainFrame::OnMouseLeftDown, this );
    Bind( wxEVT_LEFT_UP, &MainFrame::OnMouseLeftUp, this );
    Bind( wxEVT_MOTION, &MainFrame::OnMouseMove, this );
    Bind( wxEVT_SHOW, &MainFrame::OnShow, this );
    Bind( wxEVT_LEAVE_WINDOW, &MainFrame::OnMouseLeave, this );

    // Parameter connections
    auto cameraParameterCallback = [this]()
    {
      const float fov( util::FromString<uint32_t>( static_cast<const char*>( mParameterControls[3]->GetValue().utf8_str() ) ) );
      const float focalLength( util::FromString<uint32_t>( static_cast<const char*>( mParameterControls[4]->GetValue().utf8_str() ) ) );
      const float aperture( util::FromString<uint32_t>( static_cast<const char*>( mParameterControls[5]->GetValue().utf8_str() ) ) );
      mRayTracer->SetCameraParameters( fov, focalLength, aperture );
      RequestRender();
    };

    mParameterControls[2]->SetOnMouseWheelCallback( [this]() { RequestRender(); } );
    mParameterControls[3]->SetOnMouseWheelCallback( cameraParameterCallback );
    mParameterControls[4]->SetOnMouseWheelCallback( cameraParameterCallback );
    mParameterControls[5]->SetOnMouseWheelCallback( cameraParameterCallback );
  }
  catch( const std::exception& e )
  {
    logger::Logger::Instance() << "Error: " << e.what() << "\n";
  }
}

void MainFrame::RequestRender()
{
  try
  {
    const uint32_t sampleCount( util::FromString<uint32_t>( static_cast<const char*>( mParameterControls[2]->GetValue().utf8_str() ) ) );

    GLCanvas::CudaResourceGuard cudaGuard( *mGLCanvas );
    mRayTracer->Trace( cudaGuard.GetDevicePtr(), sampleCount );
    mGLCanvas->Update();
  }
  catch( const std::exception& e )
  {
    logger::Logger::Instance() << "Error: " << e.what() << "\n";
  }
}

void MainFrame::OnResizeButton( wxCommandEvent& /*event*/ )
{
  try
  {
    // Apply new settings if needed
    const math::uvec2 newImageSize( util::FromString<uint32_t>( static_cast<const char*>( mParameterControls[0]->GetValue().utf8_str() ) )
                                 , util::FromString<uint32_t>( static_cast<const char*>( mParameterControls[1]->GetValue().utf8_str() ) ) );
    if ( mGLCanvas->ImageSize() == newImageSize )
    {
      return;
    }

    const uint32_t sampleCount( util::FromString<uint32_t>( static_cast<const char*>( mParameterControls[3]->GetValue().utf8_str() ) ) );

    mGLCanvas->Resize( newImageSize );
    mRayTracer->Resize( newImageSize );

    // Rerender frame with the new size
    RequestRender();
  }
  catch( const std::exception& e )
  {
    logger::Logger::Instance() << "Error: " << e.what() << "\n";
  }
}

void MainFrame::OnRenderButton( wxCommandEvent& /*event*/ )
{
  RequestRender();
}

void MainFrame::OnStopButton( wxCommandEvent& /*event*/ )
{}

void MainFrame::OnSaveButton( wxCommandEvent& /*event*/ )
{
  try
  {
    wxFileDialog dlg( this, "Select file", wxEmptyString, "image01", "Bmp|*.bmp", wxFD_SAVE | wxFD_OVERWRITE_PROMPT );
    if ( dlg.ShowModal() == wxID_CANCEL )
    {
      return;
    }

    const wxString path( wxFileName( dlg.GetPath() ).GetFullPath() );

    // copy pixel data from GPU to CPU then write to disc
    const size_t pixelCount( mGLCanvas->ImageSize().x * mGLCanvas->ImageSize().y ) ;
    std::vector<rt::color_t> hostMem( pixelCount, 0 );

    GLCanvas::CudaResourceGuard cudaGuard( *mGLCanvas );
    cudaError_t err( cudaMemcpy( &hostMem.front(), cudaGuard.GetDevicePtr(), pixelCount * sizeof( rt::color_t ), cudaMemcpyDeviceToHost ) );
    if ( err != cudaSuccess )
    {
      throw std::runtime_error( std::string( "cannot copy pixel data from device to host: " ) + cudaGetErrorString( err ) );
    }

    rt::Bitmap bmp( mGLCanvas->ImageSize(), hostMem );
    bmp.Write( path.ToStdString() );
    logger::Logger::Instance() << "Saved to " << path.ToStdString() << "\n";
  }
  catch( const std::exception& e )
  {
    logger::Logger::Instance() << "Error: " << e.what() << "\n";
  }
}

void MainFrame::OnLogMessage( const std::string& msg )
{
  mLogTextBox->WriteText( msg );
}

void MainFrame::OnMouseMove( wxMouseEvent& event )
{
  if ( mCameraModeActive )
  {
    const math::vec2 screenPos( event.GetX(), event.GetY() );
    
    // calculate angle along x and y axes 
    // 
    // using the two screen space vectors to calculate the angle between them is not working as
    // the vectors origin is at [0, 0] and the screen left side produce`s bigger angles than the
    // right side. This is due to the triangles formed by the three points (origin, prev, current) are different!
    //   const float dot = glm::dot( glm::normalize( screenPos ), glm::normalize( mPreviousMouseScreenPosition ) );
    //    const float a = glm::acos( dot );
    //
    // so instead we work with distances as they remain the same in every part of the screen.
    // the idea is the following: assign an angle to the screen's dimensions. For example 180 deg for the full screen width.
    // let say the width is 100 pixel then 1 pixel mouse movement yields 180/100 degree angle
    
    // user defined and precomputed values
    const math::vec2 clientSize( static_cast<float>( GetClientSize().x, static_cast<float>( GetClientSize().y ) ) );
    const math::vec2 anglePerAxes( 180.0f, 180.0f );
    const math::vec2 anglePerPixel( anglePerAxes / clientSize );

    // calculations for every event
    const math::vec2 delta = mPreviousMouseScreenPosition - screenPos;
    const math::vec2 angle( anglePerPixel * delta * math::vec2( 1.0, -1.0 ) );// TODO mouse settings -> invert Y
    logger::Logger::Instance() << "MainFrame::OnMouseMove delta: " << delta << ", angle: " << angle << "\n"; 

    // apply on the raytracer camera
    mRayTracer->RotateCamera( glm::radians( angle ) );

    // request a new render from the tracer with the current parameters
    RequestRender();

    mPreviousMouseScreenPosition = screenPos;
  }
}

void MainFrame::OnMouseLeftDown( wxMouseEvent& event )
{
  mPreviousMouseScreenPosition = math::vec2( static_cast<float>( event.GetX() ), static_cast<float>( event.GetY() ) );
  mCameraModeActive = true;
}

void MainFrame::OnMouseLeftUp( wxMouseEvent& /*event*/ )
{
  mCameraModeActive = false;
}

void MainFrame::OnMouseLeave( wxMouseEvent& /*event*/ )
{
  mCameraModeActive = false;
}

void MainFrame::OnShow( wxShowEvent& event )
{
  if ( event.IsShown() )
  {
    RequestRender();
  }
}
