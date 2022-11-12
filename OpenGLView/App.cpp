#include <iostream>
#include <wx/cmdline.h>

#include "App.h"
#include "MainFrame.h"

App::App()
  : mMainFrame( std::make_unique<MainFrame>( nullptr, std::wstring( L"CudaGL Tracer 0.0.0" ), wxDefaultPosition, wxSize{ 1000, 800 } ) )
{}

App::~App()
{
  // TODO seems destruction order issue.. 
  // without this the destructor asserts in struct default_delete - default deleter for unique_ptr
  mMainFrame.release(); 
}

bool App::OnInit()
{
  wxApp::OnInit();
  return mMainFrame->Show( true );
}
