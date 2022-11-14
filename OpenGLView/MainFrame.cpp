#include <sstream>

#include "MainFrame.h"
#include "RayTracer\RayTracer.h"
#include "Common\HostUtils.h"

MainFrame::MainFrame( wxWindow* parent
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

  const math::uvec2 imageSize( static_cast<uint32_t>( 3840 / 100 )
                               , static_cast<uint32_t>( 2160 / 100 ) );
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

  mHeightEdit = new wxTextCtrl( btnPanel, wxID_ANY );
  mHeightEdit->SetValue( util::ToString( imageSize.y ) );

  mLogTextBox->SetBackgroundColour( wxColor( 125, 125, 125 ) );
  mLogTextBox->SetForegroundColour( wxColor( 200, 200, 200 ) );
  btnPanel->SetBackgroundColour( wxColor( 111, 111, 111 ) );

  resetBtn->Bind( wxEVT_COMMAND_BUTTON_CLICKED, &MainFrame::OnResizeButton, this );
  startBtn->Bind( wxEVT_COMMAND_BUTTON_CLICKED, &MainFrame::OnRenderButton, this );
  stopBtn->Bind( wxEVT_COMMAND_BUTTON_CLICKED, &MainFrame::OnStopButton, this );

  wxBoxSizer* btnSizer( new wxBoxSizer( wxVERTICAL ) );
  btnSizer->Add( mWidthEdit, 0, wxEXPAND );
  btnSizer->Add( mHeightEdit, 0, wxEXPAND );
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
  mGLCanvas->Resize( imageSize );
  
  // Rerender frame with the new settings
  mRayTracer->Trace( mGLCanvas->GetRenderTarget(), mGLCanvas->ImageSize() );
  mGLCanvas->ReleaseRenderTarget(); // TODO not safe, do better to sync this with GetRenderTarget call
  
  mGLCanvas->Update();
}

void MainFrame::OnRenderButton( wxCommandEvent& /*event*/ )
{
  mRayTracer->Trace( mGLCanvas->GetRenderTarget(), mGLCanvas->ImageSize() );
  mGLCanvas->ReleaseRenderTarget(); // TODO not safe, do better to sync this with GetRenderTarget call

  mGLCanvas->Update();
}

void MainFrame::OnStopButton( wxCommandEvent& /*event*/ )
{}
