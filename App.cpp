#include <iostream>
#include <wx/cmdline.h>

#include "Util.h"
#include "OpenGL\Canvas.h"
#include "MainFrame.h"
#include "App.h"
#include "Logger.h"

bool App::OnInit()
{
  logger::Logger::Instance() << __FUNCTION__ << "\n";

  wxApp::OnInit();

  Bind( wxEVT_KEY_DOWN, &App::OnKey, this );

  mMainFrame = new MainFrame( mTextureSize, nullptr, L"CudaGL RayTracer - test", wxDefaultPosition, { 1000, 800 } );
  return mMainFrame->Show(true);
}

int App::OnExit()
{
  logger::Logger::Instance() << __FUNCTION__ << "\n";

  mMainFrame->GetAccessible();
  return 1;
}

void App::OnKey( wxKeyEvent& event )
{
  if ( event.GetKeyCode() == WXK_ESCAPE )
  {
    Exit();
  }
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

  parser.SetDesc(cmdLineDesc);
}

bool App::OnCmdLineParsed( wxCmdLineParser& parser )
{
  wxApp::OnCmdLineParsed( parser );

  // TODO better default
  mTextureSize = math::uvec2(128, 64);

  long parsedOption = 0;
  if ( parser.Found( wxT("rx"), &parsedOption ) )
  {
    mTextureSize.x = static_cast<uint32_t>( parsedOption );
  }

  parsedOption = 0;
  if ( parser.Found( wxT("ry"), &parsedOption ) )
  {
    mTextureSize.y = static_cast<uint32_t>( parsedOption );
  }

  return true;
}
