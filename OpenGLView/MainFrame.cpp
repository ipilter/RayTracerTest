#include <sstream>

#include "MainFrame.h"
#include "RayTracer\RayTracer.h"

MainFrame::MainFrame( wxWindow* parent
                      , std::wstring title
                      , const wxPoint& pos
                      , const wxSize& size )
  : wxFrame( parent, wxID_ANY, title, pos, size )
  , mRayTracer( std::make_unique<rt::RayTracer>() )
{

  // dummy stuff to test the libraries
  auto* mainPanel = new wxPanel( this, wxID_ANY );

  mGLCanvas = std::make_unique<GLCanvas>( math::uvec2( 1, 1 ), mainPanel, wxID_ANY );

  rt::RenderData renderData;

  renderData.mPixelBuffer = nullptr; //  mGLCanvas->GetBackBuffer();
  renderData.mDimensions = mGLCanvas->ImageSize();

  mRayTracer->Trace( renderData );
}

MainFrame::~MainFrame()
{}
