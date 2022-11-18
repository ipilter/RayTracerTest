#include <sstream>

#include "MainFrame.h"
#include "RayTracer\RayTracer.h"
#include "Common\HostUtils.h"

MainFrame::MainFrame( const math::uvec2& imageSize
                      , wxWindow* parent
                      , std::wstring title
                      , const wxPoint& pos
                      , const wxSize& size )
  : wxFrame( parent, wxID_ANY, title, pos, size )
  , mGLCanvas( nullptr )
  , mLogTextBox( nullptr )
  , mRayTracer( std::make_unique<rt::RayTracer>() )
{
  wxPanel* mainPanel( new wxPanel( this, wxID_ANY ) );
  mLogTextBox = new wxTextCtrl( mainPanel
                                , wxID_ANY
                                , wxEmptyString
                                , wxDefaultPosition
                                , wxDefaultSize
                                , wxTE_MULTILINE );

  mGLCanvas = std::make_unique<GLCanvas>( imageSize, mainPanel, wxID_ANY );

  wxBoxSizer* mainSizer( new wxBoxSizer( wxVERTICAL ) );
  mainSizer->Add( mGLCanvas.get(), 90, wxEXPAND );
  mainSizer->Add( mLogTextBox, 10, wxEXPAND );
  mainPanel->SetSizer( mainSizer );

  wxPanel* btnPanel( new wxPanel( this, wxID_ANY ) );
  wxButton* resetBtn( new wxButton( btnPanel, wxID_ANY, "Resize" ) );
  wxButton* startBtn( new wxButton( btnPanel, wxID_ANY, "Render" ) );
  wxButton* stopBtn( new wxButton( btnPanel, wxID_ANY, "Stop" ) );

  mWidthEdit = new wxTextCtrl( btnPanel, wxID_ANY );
  mWidthEdit->SetValue( util::ToString( imageSize.x ) );
  wxStaticText* widthLabel = new wxStaticText( btnPanel, wxID_ANY, "Width" );
  wxBoxSizer* widthSizer( new wxBoxSizer( wxHORIZONTAL ) );
  widthSizer->Add(widthLabel);
  widthSizer->AddStretchSpacer();
  widthSizer->Add(mWidthEdit);

  mHeightEdit = new wxTextCtrl( btnPanel, wxID_ANY );
  mHeightEdit->SetValue( util::ToString( imageSize.y ) );
  wxStaticText* heightLabel = new wxStaticText( btnPanel, wxID_ANY, "Height" );
  wxBoxSizer* heightSizer( new wxBoxSizer( wxHORIZONTAL ) );
  heightSizer->Add(heightLabel);
  heightSizer->AddStretchSpacer();
  heightSizer->Add(mHeightEdit);

  mSampleCountEdit = new wxTextCtrl( btnPanel, wxID_ANY );
  mSampleCountEdit->SetValue( util::ToString( 8 ) );
  wxStaticText* sampleLabel = new wxStaticText( btnPanel, wxID_ANY, "Samples" );
  wxBoxSizer* sampleSizer( new wxBoxSizer( wxHORIZONTAL ) );
  sampleSizer->Add(sampleLabel);
  sampleSizer->AddStretchSpacer();
  sampleSizer->Add(mSampleCountEdit);

  mFovEdit = new wxTextCtrl( btnPanel, wxID_ANY );
  mFovEdit->SetValue( util::ToString( 90 ) );
  wxStaticText* fovLabel = new wxStaticText( btnPanel, wxID_ANY, "Fov" );
  wxBoxSizer* fovSizer( new wxBoxSizer( wxHORIZONTAL ) );
  fovSizer->Add(fovLabel);
  fovSizer->AddStretchSpacer();
  fovSizer->Add(mFovEdit);

  mFocalLengthEdit = new wxTextCtrl( btnPanel, wxID_ANY );
  mFocalLengthEdit->SetValue( util::ToString( 50 ) );
  wxStaticText* focalLengthLabel = new wxStaticText( btnPanel, wxID_ANY, "Focal l." );
  wxBoxSizer* focalLengthSizer( new wxBoxSizer( wxHORIZONTAL ) );
  focalLengthSizer->Add(focalLengthLabel);
  focalLengthSizer->AddStretchSpacer();
  focalLengthSizer->Add(mFocalLengthEdit);

  mApertureEdit = new wxTextCtrl( btnPanel, wxID_ANY );
  mApertureEdit->SetValue( util::ToString( 1.7 ) );
  wxStaticText* apertureLabel = new wxStaticText( btnPanel, wxID_ANY, "Aperture" );
  wxBoxSizer* apertureSizer( new wxBoxSizer( wxHORIZONTAL ) );
  apertureSizer->Add(apertureLabel);
  apertureSizer->AddStretchSpacer(2);
  apertureSizer->Add(mApertureEdit);

  mLogTextBox->SetBackgroundColour( wxColor( 125, 125, 125 ) );
  mLogTextBox->SetForegroundColour( wxColor( 200, 200, 200 ) );
  btnPanel->SetBackgroundColour( wxColor( 111, 111, 111 ) );

  resetBtn->Bind( wxEVT_COMMAND_BUTTON_CLICKED, &MainFrame::OnResizeButton, this );
  startBtn->Bind( wxEVT_COMMAND_BUTTON_CLICKED, &MainFrame::OnRenderButton, this );
  stopBtn->Bind( wxEVT_COMMAND_BUTTON_CLICKED, &MainFrame::OnStopButton, this );

  wxBoxSizer* btnSizer( new wxBoxSizer( wxVERTICAL ) );
  btnSizer->Add( widthSizer, 0, wxEXPAND );
  btnSizer->Add( heightSizer, 0, wxEXPAND );
  btnSizer->Add( sampleSizer, 0, wxEXPAND );  

  btnSizer->Add( fovSizer, 0, wxEXPAND );  
  btnSizer->Add( focalLengthSizer, 0, wxEXPAND );  
  btnSizer->Add( apertureSizer, 0, wxEXPAND );  

  btnSizer->Add( resetBtn, 0, wxEXPAND );
  btnSizer->Add( startBtn, 0, wxEXPAND );
  btnSizer->Add( stopBtn, 0, wxEXPAND );
  btnPanel->SetSizer( btnSizer );

  wxBoxSizer* sizer( new wxBoxSizer( wxHORIZONTAL ) );
  sizer->Add( mainPanel, 1, wxEXPAND );
  sizer->Add( btnPanel, 0, wxEXPAND );

  this->SetSizer( sizer );
}

MainFrame::~MainFrame()
{}

void MainFrame::OnResizeButton( wxCommandEvent& /*event*/ )
{
  // Apply new settings
  const math::uvec2 imageSize( util::FromString<uint32_t>( static_cast<const char*>( mWidthEdit->GetValue().utf8_str() ) )
                               , util::FromString<uint32_t>( static_cast<const char*>( mHeightEdit->GetValue().utf8_str() ) ) );
  const uint32_t sampleCount( util::FromString<uint32_t>( static_cast<const char*>( mSampleCountEdit->GetValue().utf8_str() ) ) );
  const float fov( util::FromString<uint32_t>( static_cast<const char*>( mFovEdit->GetValue().utf8_str() ) ) );
  const float focalLength( util::FromString<uint32_t>( static_cast<const char*>( mFocalLengthEdit->GetValue().utf8_str() ) ) );
  const float aperture( util::FromString<uint32_t>( static_cast<const char*>( mApertureEdit->GetValue().utf8_str() ) ) );

  mGLCanvas->Resize( imageSize );
  mRayTracer->Resize( imageSize );
  
  // Rerender frame with the new settings
  mRayTracer->Trace( mGLCanvas->GetRenderTarget(), mGLCanvas->ImageSize(), sampleCount, fov, focalLength, aperture );
  mGLCanvas->ReleaseRenderTarget(); // TODO not safe, do better to sync this with GetRenderTarget call
  
  mGLCanvas->Update();
}

void MainFrame::OnRenderButton( wxCommandEvent& /*event*/ )
{
  const uint32_t sampleCount( util::FromString<uint32_t>( static_cast<const char*>( mSampleCountEdit->GetValue().utf8_str() ) ) );
  const float fov( util::FromString<uint32_t>( static_cast<const char*>( mFovEdit->GetValue().utf8_str() ) ) );
  const float focalLength( util::FromString<uint32_t>( static_cast<const char*>( mFocalLengthEdit->GetValue().utf8_str() ) ) );
  const float aperture( util::FromString<uint32_t>( static_cast<const char*>( mApertureEdit->GetValue().utf8_str() ) ) );
  mRayTracer->Trace( mGLCanvas->GetRenderTarget(), mGLCanvas->ImageSize(), sampleCount, fov, focalLength, aperture );
  mGLCanvas->ReleaseRenderTarget(); // TODO not safe, do better to sync this with GetRenderTarget call

  mGLCanvas->Update();
}

void MainFrame::OnStopButton( wxCommandEvent& /*event*/ )
{}

