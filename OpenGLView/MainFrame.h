#pragma once

#include "GLCanvas.h"
#include "RayTracer\RayTracer.h"

class MainFrame : public wxFrame, public virtual ISptr<MainFrame>
{
public:
  MainFrame( const math::uvec2& imageSize
             , wxWindow* parent
             , std::wstring title
             , const wxPoint& pos
             , const wxSize& size );

  virtual ~MainFrame();

private:
  void OnResizeButton( wxCommandEvent& event );
  void OnRenderButton( wxCommandEvent& event );
  void OnStopButton( wxCommandEvent& event );

  GLCanvas::uptr mGLCanvas;
  wxTextCtrl* mLogTextBox;
  wxTextCtrl* mWidthEdit;
  wxTextCtrl* mHeightEdit;

  rt::RayTracer::uptr mRayTracer;
};
