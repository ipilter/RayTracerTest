#include <wx/colordlg.h>
#include <sstream>

#include "MainFrame.h"
#include "OpenGL/Canvas.h"

#include "Logger.h"

MainFrame::MainFrame( const math::uvec2& textureSize
                      , wxWindow* parent
                      , std::wstring title
                      , const wxPoint& pos
                      , const wxSize& size )
  : wxFrame( parent, wxID_ANY, title, pos, size )
  , mGLCanvas( nullptr )
  , mLogTextBox( nullptr )
{
  logger::Logger::Instance() << __FUNCTION__ << "\n";

  auto* mainPanel = new wxPanel( this, wxID_ANY );
  mLogTextBox = new wxTextCtrl( mainPanel, wxID_ANY
                                , wxEmptyString
                                , wxDefaultPosition
                                , wxDefaultSize
                                , wxTE_MULTILINE );
  mGLCanvas = new gl::Canvas( textureSize, mainPanel, wxID_ANY );

  auto* mainSizer = new wxBoxSizer( wxVERTICAL );
  mainSizer->Add( mGLCanvas, 90, wxEXPAND );
  mainSizer->Add( mLogTextBox, 10, wxEXPAND );
  mainPanel->SetSizer( mainSizer );

  mLogTextBox->SetBackgroundColour( wxColor( 21, 21, 21 ) );
  mLogTextBox->SetForegroundColour( wxColor( 180, 180, 180 ) );

  auto* btnPanel = new wxPanel( this, wxID_ANY );
  auto* startBtn = new wxButton( btnPanel, wxID_ANY, "Start" );
  auto* stopBtn = new wxButton( btnPanel, wxID_ANY, "Stop" );

  btnPanel->SetBackgroundColour( wxColor( 21, 21, 21 ) );

  startBtn->Bind(wxEVT_COMMAND_BUTTON_CLICKED, &MainFrame::OnStartButton, this);
  stopBtn->Bind(wxEVT_COMMAND_BUTTON_CLICKED, &MainFrame::OnStopButton, this);

  auto* btnSizer = new wxBoxSizer( wxVERTICAL );
  btnSizer->Add( startBtn, 0, wxEXPAND );
  btnSizer->Add( stopBtn, 0, wxEXPAND );
  btnPanel->SetSizer( btnSizer );

  auto* sizer = new wxBoxSizer( wxHORIZONTAL );
  sizer->Add( mainPanel, 1, wxEXPAND );
  sizer->Add( btnPanel, 0, wxEXPAND );

  this->SetSizer( sizer );

  mRayTracer = std::make_shared<rt::RayTracer>();
}

MainFrame::~MainFrame()
{
  logger::Logger::Instance() << __FUNCTION__ << "\n";
}

void MainFrame::AddLogMessage( const std::string& msg )
{
  mLogTextBox->WriteText( ( msg + "\n" ) );
}

void MainFrame::OnStartButton(wxCommandEvent& event)
{
  mRayTracer->Trace( mGLCanvas->GetRenderTarget(), mGLCanvas->GetTextureSize() );
  mGLCanvas->UpdateTexture();
}

void MainFrame::OnStopButton(wxCommandEvent& event)
{
}
