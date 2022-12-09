#pragma once

#include "GLCanvas.h"
#include "WxNamedTextControl.h"
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
  void InitializeUIElements();

  void OnResizeButton( wxCommandEvent& event );
  void OnRenderButton( wxCommandEvent& event );
  void OnStopButton( wxCommandEvent& event );
  void OnSaveButton( wxCommandEvent& event );

  wxPanel* mMainPanel;
  wxPanel* mControlPanel;
  wxTextCtrl* mLogTextBox;

  // TODO lazy string as key
  std::unordered_map<std::string, wxNamedTextControl*> mParameterControls;
  using EventCallBack = std::function<void( wxCommandEvent& )>;
  std::unordered_map<std::string, std::pair<wxButton*, EventCallBack>> mButtons;

  GLCanvas::uptr mGLCanvas;
  rt::RayTracer::uptr mRayTracer;
};
