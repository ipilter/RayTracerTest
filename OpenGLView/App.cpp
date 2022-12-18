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
  , mCameraPosition( 0.0f )
  , mCameraAngles( 0.0f )
  , mFov( 70.0f )
  , mFocalLength( 50.0f )
  , mAperture( 4.5f )
  , mAnglesPerAxes( 180.0f, 180.0f )
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
                                            , mCameraPosition
                                            , mCameraAngles
                                            , mFov
                                            , mFocalLength
                                            , mAperture
                                            , mAnglesPerAxes
                                            , nullptr
                                            , std::wstring( L"CudaGL Tracer 0.0.1" )
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
    { wxCMD_LINE_OPTION, "h", "height", "render height (pixels)", wxCMD_LINE_VAL_NUMBER },
    { wxCMD_LINE_OPTION, "s", "samples", "samples per pixel", wxCMD_LINE_VAL_NUMBER },

    { wxCMD_LINE_OPTION, "cx", "camerax", "camera position x", wxCMD_LINE_VAL_NUMBER },
    { wxCMD_LINE_OPTION, "cy", "cameray", "camera position y", wxCMD_LINE_VAL_NUMBER },
    { wxCMD_LINE_OPTION, "cz", "cameraz", "camera position z", wxCMD_LINE_VAL_NUMBER },

    { wxCMD_LINE_OPTION, "cxa", "cameraxangle", "camera angle on (1,0,0) axis", wxCMD_LINE_VAL_NUMBER },
    { wxCMD_LINE_OPTION, "cya", "camerayangle", "camera angle on (0,1,0) axis", wxCMD_LINE_VAL_NUMBER },

    { wxCMD_LINE_OPTION, "f", "fov", "field of view", wxCMD_LINE_VAL_NUMBER },
    { wxCMD_LINE_OPTION, "l", "focallength", "focal length", wxCMD_LINE_VAL_NUMBER },
    { wxCMD_LINE_OPTION, "a", "aperture", "aperture", wxCMD_LINE_VAL_NUMBER },

    { wxCMD_LINE_OPTION, "apw", "anglesperwidth", "angles per view width", wxCMD_LINE_VAL_NUMBER },
    { wxCMD_LINE_OPTION, "aph", "anglesperheight", "angles per view height", wxCMD_LINE_VAL_NUMBER },

    { wxCMD_LINE_NONE }
  };

  parser.SetDesc( cmdLineDesc );
}

bool App::OnCmdLineParsed( wxCmdLineParser& parser )
{
  wxApp::OnCmdLineParsed( parser );

  // image dimensions
  long parsedOption = 0;
  if ( parser.Found( wxT( "w" ), &parsedOption ) )
  {
    mTextureSize.x = static_cast<uint32_t>( parsedOption );
  }

  if ( parser.Found( wxT( "h" ), &parsedOption ) )
  {
    mTextureSize.y = static_cast<uint32_t>( parsedOption );
  }

  // samples per pixel
  if ( parser.Found( wxT( "s" ), &parsedOption ) )
  {
    mSampleCount = static_cast<uint32_t>( parsedOption );
  }

  // camera position
  if ( parser.Found( wxT( "cx" ), &parsedOption ) )
  {
    mCameraPosition.x = static_cast<float>( parsedOption );
  }

  if ( parser.Found( wxT( "cy" ), &parsedOption ) )
  {
    mCameraPosition.y = static_cast<float>( parsedOption );
  }

  if ( parser.Found( wxT( "cz" ), &parsedOption ) )
  {
    mCameraPosition.z = static_cast<float>( parsedOption );
  }

  // camera orientation
  if ( parser.Found( wxT( "cxa" ), &parsedOption ) )
  {
    mCameraAngles.x = glm::radians( static_cast<float>( parsedOption ) );
  }

  if ( parser.Found( wxT( "cya" ), &parsedOption ) )
  {
    mCameraAngles.y = glm::radians( static_cast<float>( parsedOption ) );
  }
  
  // camera parameters
  if ( parser.Found( wxT( "f" ), &parsedOption ) )
  {
    mFov = static_cast<float>( parsedOption );
  }

  if ( parser.Found( wxT( "l" ), &parsedOption ) )
  {
    mFocalLength = static_cast<float>( parsedOption );
  }

  if ( parser.Found( wxT( "a" ), &parsedOption ) )
  {
    mAperture = static_cast<float>( parsedOption );
  }

  // angles per view size
  if ( parser.Found( wxT( "apw" ), &parsedOption ) )
  {
    mAnglesPerAxes.x = static_cast<float>( parsedOption );
  }

  if ( parser.Found( wxT( "aph" ), &parsedOption ) )
  {
    mAnglesPerAxes.y = static_cast<float>( parsedOption );
  }

  return true;
}
