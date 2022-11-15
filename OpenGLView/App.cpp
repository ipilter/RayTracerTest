#include <iostream>
#include <wx/cmdline.h>

#include "App.h"
#include "MainFrame.h"

App::App()
  : mMainFrame( nullptr )
  , mTextureSize( 3840 / 100, 2160 / 100 )
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

  mMainFrame = std::make_unique<MainFrame>( mTextureSize
                                            , nullptr
                                            , std::wstring( L"CudaGL Tracer 0.0.0" )
                                            , wxDefaultPosition
                                            , wxSize{ 1000, 800 } );
  return mMainFrame->Show( true );
}

int App::OnExit()
{
  return 0;
}

void App::OnInitCmdLine( wxCmdLineParser& parser )
{
  static const wxCmdLineEntryDesc cmdLineDesc[] =
  {
    { wxCMD_LINE_SWITCH, "v", "verbose", "be verbose" },
    { wxCMD_LINE_SWITCH, "q", "quiet",   "be quiet" },
    { wxCMD_LINE_OPTION, "rx", "resx", "x resolution", wxCMD_LINE_VAL_NUMBER },
    { wxCMD_LINE_OPTION, "ry", "resx", "y resolution", wxCMD_LINE_VAL_NUMBER },
    { wxCMD_LINE_NONE }
  };

  parser.SetDesc( cmdLineDesc );
}

bool App::OnCmdLineParsed( wxCmdLineParser& parser )
{
  wxApp::OnCmdLineParsed( parser );

  long parsedOption = 0;
  if ( parser.Found( wxT( "rx" ), &parsedOption ) )
  {
    mTextureSize.x = static_cast<uint32_t>( parsedOption );
  }

  parsedOption = 0;
  if ( parser.Found( wxT( "ry" ), &parsedOption ) )
  {
    mTextureSize.y = static_cast<uint32_t>( parsedOption );
  }

  return true;
}
