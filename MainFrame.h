#pragma once

#include "WxMain.h"
#include "Math.h"
#include "RayTracer/RayTracer.h"

namespace gl
{
class Canvas;
}

class MainFrame : public wxFrame
{
public:
  MainFrame( const math::uvec2& textureSize
             , wxWindow* parent
             , std::wstring title
             , const wxPoint& pos
             , const wxSize& size );

  virtual ~MainFrame();

  void AddLogMessage( const std::string& msg );

private:
  void OnStartButton( wxCommandEvent& event );
  void OnStopButton( wxCommandEvent& event );

private:
  rt::RayTracer::sptr mRayTracer;
  gl::Canvas* mGLCanvas;
  wxTextCtrl* mLogTextBox;
};
