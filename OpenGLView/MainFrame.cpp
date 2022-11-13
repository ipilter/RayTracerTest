#include <sstream>

#include "MainFrame.h"
#include "RayTracer\RayTracer.h"

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
  mGLCanvas = std::make_unique<GLCanvas>( math::uvec2( static_cast<uint32_t>( 1920/100 )
                                                       , static_cast<uint32_t>( 1080/100 ) ), mainPanel, wxID_ANY ); // glm::pow( 2, 4 )

  wxBoxSizer* mainSizer( new wxBoxSizer( wxVERTICAL ) );
  mainSizer->Add( mGLCanvas.get(), 90, wxEXPAND );
  mainSizer->Add( mLogTextBox, 10, wxEXPAND );
  mainPanel->SetSizer( mainSizer );

  wxPanel* btnPanel( new wxPanel( this, wxID_ANY ) );
  wxButton* startBtn( new wxButton( btnPanel, wxID_ANY, "Start" ) );
  wxButton* stopBtn( new wxButton( btnPanel, wxID_ANY, "Stop" ) );

  mWidthEdit = new wxTextCtrl( btnPanel, wxID_ANY );
  mWidthEdit->SetValue( "1024" );

  mHeightEdit = new wxTextCtrl( btnPanel, wxID_ANY );
  mHeightEdit->SetValue( "1024" );

  mLogTextBox->SetBackgroundColour( wxColor( 125, 125, 125 ) );
  mLogTextBox->SetForegroundColour( wxColor( 200, 200, 200 ) );
  btnPanel->SetBackgroundColour( wxColor( 111, 111, 111 ) );

  startBtn->Bind( wxEVT_COMMAND_BUTTON_CLICKED, &MainFrame::OnStartButton, this );
  stopBtn->Bind( wxEVT_COMMAND_BUTTON_CLICKED, &MainFrame::OnStopButton, this );

  wxBoxSizer* btnSizer( new wxBoxSizer( wxVERTICAL ) );
  btnSizer->Add( mWidthEdit, 0, wxEXPAND );
  btnSizer->Add( mHeightEdit, 0, wxEXPAND );
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

void MainFrame::OnStartButton( wxCommandEvent& /*event*/ )
{
  rt::RenderData renderData( mGLCanvas->GetRenderTarget(), mGLCanvas->ImageSize() );
  mRayTracer->Trace( renderData );
  mGLCanvas->ReleaseRenderTarget(); // TODO not safe, do better

  mGLCanvas->UpdateTexture();
}

void MainFrame::OnStopButton( wxCommandEvent& /*event*/ )
{}
