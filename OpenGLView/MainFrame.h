#pragma once

#include "GLCanvas.h"
#include "RayTracer\RayTracer.h"

class MainFrame : public wxFrame, public virtual ISptr<MainFrame>
{
public:
  MainFrame( wxWindow* parent
             , std::wstring title
             , const wxPoint& pos
             , const wxSize& size );

  virtual ~MainFrame();

private:
  void OnStartButton( wxCommandEvent& event );
  void OnStopButton( wxCommandEvent& event );

  GLCanvas::uptr mGLCanvas;
  wxTextCtrl* mLogTextBox;

  rt::RayTracer::uptr mRayTracer;
};
