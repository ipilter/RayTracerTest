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
  mGLCanvas = std::make_unique<GLCanvas>( math::uvec2( 1, 1 ), mainPanel, wxID_ANY );

  wxBoxSizer* mainSizer( new wxBoxSizer( wxVERTICAL ) );
  mainSizer->Add( mGLCanvas.get(), 90, wxEXPAND );
  mainSizer->Add( mLogTextBox, 10, wxEXPAND );
  mainPanel->SetSizer( mainSizer );

  wxPanel* btnPanel( new wxPanel( this, wxID_ANY ) );
  wxButton* startBtn( new wxButton( btnPanel, wxID_ANY, "Start" ) );
  wxButton* stopBtn( new wxButton( btnPanel, wxID_ANY, "Stop" ) );

  mLogTextBox->SetBackgroundColour( wxColor( 225, 225, 225 ) );
  mLogTextBox->SetForegroundColour( wxColor( 250, 250, 250 ) );
  btnPanel->SetBackgroundColour( wxColor( 225, 225, 225 ) );

  startBtn->Bind( wxEVT_COMMAND_BUTTON_CLICKED, &MainFrame::OnStartButton, this );
  stopBtn->Bind( wxEVT_COMMAND_BUTTON_CLICKED, &MainFrame::OnStopButton, this );

  wxBoxSizer* btnSizer( new wxBoxSizer( wxVERTICAL ) );
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
  rt::RenderData renderData;

  renderData.mPixelBuffer = nullptr; //  mGLCanvas->GetBackBuffer();
  renderData.mDimensions = mGLCanvas->ImageSize();

  mRayTracer->Trace( renderData );

  //mGLCanvas->UpdateTexture();
}

void MainFrame::OnStopButton( wxCommandEvent& /*event*/ )
{}
