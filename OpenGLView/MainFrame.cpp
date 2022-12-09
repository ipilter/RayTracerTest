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

MainFrame::MainFrame( const math::uvec2& imageSize
                      , wxWindow* parent
                      , std::wstring title
                      , const wxPoint& pos
                      , const wxSize& size )
  : wxFrame( parent, wxID_ANY, title, pos, size )
  , mMainPanel( new wxPanel( this, wxID_ANY ) )
  , mControlPanel( new wxPanel( this, wxID_ANY ) )
  , mLogTextBox( new wxTextCtrl( mMainPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxTE_MULTILINE ) )
  , mGLCanvas( std::make_unique<GLCanvas>( imageSize, mMainPanel, wxID_ANY ) )
  , mRayTracer( std::make_unique<rt::RayTracer>( imageSize ) )
{
  // Parameters
  mParameterControls["Width"] = new wxNamedTextControl(mControlPanel, wxID_ANY, "Width", util::ToString(imageSize.x));
  mParameterControls["Height"] = new wxNamedTextControl( mControlPanel, wxID_ANY, "Height", util::ToString( imageSize.y ) );
  mParameterControls["Samples"] = new wxNamedTextControl( mControlPanel, wxID_ANY, "Samples", util::ToString( 32 ) );
  mParameterControls["Fov"] = new wxNamedTextControl( mControlPanel, wxID_ANY, "Fov", util::ToString( 70 ) );
  mParameterControls["Focal l."] = new wxNamedTextControl( mControlPanel, wxID_ANY, "Focal l.", util::ToString( 50 ) );
  mParameterControls["Aperture"] = new wxNamedTextControl( mControlPanel, wxID_ANY, "Aperture", util::ToString( 8.0 ) );

  // Buttons
  mButtons["Resize"] = std::make_pair(new wxButton( mControlPanel, wxID_ANY, "Resize" ), std::bind( &MainFrame::OnResizeButton, this, std::placeholders::_1 ));
  mButtons["Render"] = std::make_pair(new wxButton( mControlPanel, wxID_ANY, "Render" ), std::bind( &MainFrame::OnRenderButton, this, std::placeholders::_1 ));
  mButtons["Stop"] = std::make_pair(new wxButton( mControlPanel, wxID_ANY, "Stop" ), std::bind( &MainFrame::OnStopButton, this, std::placeholders::_1 ));
  mButtons["Save"] = std::make_pair(new wxButton( mControlPanel, wxID_ANY, "Save" ), std::bind( &MainFrame::OnSaveButton, this, std::placeholders::_1 ));
  
  // TODO add some default values for them
  InitializeUIElements();
}

MainFrame::~MainFrame()
{}

void MainFrame::InitializeUIElements()
{
  wxBoxSizer* controlSizer( new wxBoxSizer( wxVERTICAL ) );
  for ( auto ctrl : mParameterControls )
  {
    controlSizer->Add( ctrl.second, 0, wxEXPAND );
  }

  for ( auto btn : mButtons )
  {
    controlSizer->Add( btn.second.first, 0, wxEXPAND );
    btn.second.first->Bind( wxEVT_COMMAND_BUTTON_CLICKED, btn.second.second );
  }
  mControlPanel->SetSizer( controlSizer );

  wxBoxSizer* mainSizer( new wxBoxSizer( wxVERTICAL ) );
  mainSizer->Add( mGLCanvas.get(), 90, wxEXPAND );
  mainSizer->Add( mLogTextBox, 10, wxEXPAND );
  mMainPanel->SetSizer( mainSizer );

  wxBoxSizer* sizer( new wxBoxSizer( wxHORIZONTAL ) );
  sizer->Add( mMainPanel, 1, wxEXPAND );
  sizer->Add( mControlPanel, 0, wxEXPAND );
  this->SetSizer( sizer );

  // Some colors
  for ( auto ctrl : mParameterControls )
  {
    ctrl.second->SetBackgroundColour( wxColor( 115, 115, 115 ) );
  }

  // TODO use common anchestor's ptr in a loop instead these
  mLogTextBox->SetBackgroundColour( wxColor( 065, 065, 065 ) );
  mLogTextBox->SetForegroundColour( wxColor( 070, 070, 070 ) );
  mControlPanel->SetBackgroundColour( wxColor( 075, 075, 075 ) );
}

void MainFrame::OnResizeButton( wxCommandEvent& /*event*/ )
{
  // Apply new settings if needed
  const math::uvec2 newImageSize( util::FromString<uint32_t>( static_cast<const char*>( mParameterControls["Width"]->GetValue().utf8_str() ) )
                               , util::FromString<uint32_t>( static_cast<const char*>( mParameterControls["Height"]->GetValue().utf8_str() ) ) );
  if ( mGLCanvas->ImageSize() == newImageSize )
  {
    return;
  }

  const uint32_t sampleCount( util::FromString<uint32_t>( static_cast<const char*>( mParameterControls["Samples"]->GetValue().utf8_str() ) ) );
  const float fov( util::FromString<uint32_t>( static_cast<const char*>( mParameterControls["Fov"]->GetValue().utf8_str() ) ) );
  const float focalLength( util::FromString<uint32_t>( static_cast<const char*>( mParameterControls["Focal l."]->GetValue().utf8_str() ) ) );
  const float aperture( util::FromString<uint32_t>( static_cast<const char*>( mParameterControls["Aperture"]->GetValue().utf8_str() ) ) );

  mGLCanvas->Resize( newImageSize );
  mRayTracer->Resize( newImageSize );

  // Rerender frame with the new size
  GLCanvas::CudaResourceGuard cudaGuard( *mGLCanvas );
  mRayTracer->Trace( cudaGuard.GetDevicePtr(), sampleCount, fov, focalLength, aperture );

  mGLCanvas->Update();
}

void MainFrame::OnRenderButton( wxCommandEvent& /*event*/ )
{
  const uint32_t sampleCount( util::FromString<uint32_t>( static_cast<const char*>( mParameterControls["Samples"]->GetValue().utf8_str() ) ) );

  const float fov( util::FromString<uint32_t>( static_cast<const char*>( mParameterControls["Fov"]->GetValue().utf8_str() ) ) );
  const float focalLength( util::FromString<uint32_t>( static_cast<const char*>( mParameterControls["Focal l."]->GetValue().utf8_str() ) ) );
  const float aperture( util::FromString<uint32_t>( static_cast<const char*>( mParameterControls["Aperture"]->GetValue().utf8_str() ) ) );

  GLCanvas::CudaResourceGuard cudaGuard( *mGLCanvas );
  mRayTracer->Trace( cudaGuard.GetDevicePtr(), sampleCount, fov, focalLength, aperture );

  mGLCanvas->Update();
}

void MainFrame::OnStopButton( wxCommandEvent& /*event*/ )
{}

void MainFrame::OnSaveButton( wxCommandEvent& /*event*/ )
{
  wxFileDialog dlg( this, "Select file", wxEmptyString, wxEmptyString, "Bmp|*.bmp", wxFD_SAVE | wxFD_OVERWRITE_PROMPT );
  dlg.ShowModal();
  wxFileName fileName = dlg.GetPath();
  wxString path = fileName.GetFullPath();

  // copy pixel data from GPU to CPU then write to disc
  auto pixelCount = mGLCanvas->ImageSize().x * mGLCanvas->ImageSize().y;
  std::vector<uint32_t> hostMem( pixelCount, 0 );

  GLCanvas::CudaResourceGuard cudaGuard( *mGLCanvas );
  cudaError_t err( cudaMemcpy( &hostMem.front(), cudaGuard.GetDevicePtr(), pixelCount * sizeof( uint32_t ), cudaMemcpyDeviceToHost ) );
  if ( err != cudaSuccess )
  {
    throw std::runtime_error( std::string( "cannot copy pixel data from device to host: " ) + cudaGetErrorString( err ) );
  }

  Bitmap bmp( mGLCanvas->ImageSize(), hostMem );
  bmp.Write( path.ToStdString() );
}
