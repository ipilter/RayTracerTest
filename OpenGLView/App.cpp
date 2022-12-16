#include <iostream>
#include <wx/cmdline.h>

#include "App.h"
#include "MainFrame.h"

#include <wx/display.h>

#include "Common\Logger.h"

App::App()
  : mMainFrame( nullptr )
  , mTextureSize( 3840 / 100, 2160 / 100 )
  , mSampleCount( 1 )
  , mFov( 70.0 )
  , mFocalLength( 50.0 )
  , mAperture( 4.5 )
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

  const wxDisplay display;

  mMainFrame = std::make_unique<MainFrame>( mTextureSize
                                            , mSampleCount
                                            , mFov
                                            , mFocalLength
                                            , mAperture
                                            , nullptr
                                            , std::wstring( L"CudaGL Tracer 0.0.0" )
                                            , wxDefaultPosition
                                            , wxSize{ static_cast<int>( display.GetGeometry().width * 0.75f )
                                                      , static_cast<int>( display.GetGeometry().height * 0.75f ) } );
  return mMainFrame->Show( true );
}

int App::OnExit()
{
  logger::Logger::Instance() << "Exiting...\n";
  return 0;
}

void App::OnInitCmdLine( wxCmdLineParser& parser )
{
  static const wxCmdLineEntryDesc cmdLineDesc[] =
  {
    { wxCMD_LINE_SWITCH, "v", "verbose", "be verbose" },
    { wxCMD_LINE_SWITCH, "q", "quiet",   "be quiet" },
    { wxCMD_LINE_OPTION, "w", "width", "render width (pixels)", wxCMD_LINE_VAL_NUMBER },
    { wxCMD_LINE_OPTION, "h", "heght", "render height (pixels)", wxCMD_LINE_VAL_NUMBER },
    { wxCMD_LINE_OPTION, "s", "samples", "samples per pixel", wxCMD_LINE_VAL_NUMBER },
    { wxCMD_LINE_OPTION, "f", "fov", "field of view", wxCMD_LINE_VAL_NUMBER },
    { wxCMD_LINE_OPTION, "l", "focallength", "focal length", wxCMD_LINE_VAL_NUMBER },
    { wxCMD_LINE_OPTION, "a", "aperture", "aperture", wxCMD_LINE_VAL_NUMBER },
    { wxCMD_LINE_NONE }
  };

  parser.SetDesc( cmdLineDesc );
}

bool App::OnCmdLineParsed( wxCmdLineParser& parser )
{
  wxApp::OnCmdLineParsed( parser );

  long parsedOption = 0;
  if ( parser.Found( wxT( "w" ), &parsedOption ) )
  {
    mTextureSize.x = static_cast<uint32_t>( parsedOption );
  }

  if ( parser.Found( wxT( "h" ), &parsedOption ) )
  {
    mTextureSize.y = static_cast<uint32_t>( parsedOption );
  }

  if ( parser.Found( wxT( "s" ), &parsedOption ) )
  {
    mSampleCount = static_cast<uint32_t>( parsedOption );
  }

  if ( parser.Found( wxT( "f" ), &parsedOption ) )
  {
    mFov = static_cast<uint32_t>( parsedOption );
  }

  if ( parser.Found( wxT( "l" ), &parsedOption ) )
  {
    mFocalLength = static_cast<uint32_t>( parsedOption );
  }

  if ( parser.Found( wxT( "a" ), &parsedOption ) )
  {
    mAperture = static_cast<uint32_t>( parsedOption );
  }
  return true;
}
