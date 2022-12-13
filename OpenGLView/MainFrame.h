#pragma once

#include <wx\splitter.h>

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
  void RequestRender();

  void OnResizeButton( wxCommandEvent& event );
  void OnRenderButton( wxCommandEvent& event );
  void OnStopButton( wxCommandEvent& event );
  void OnSaveButton( wxCommandEvent& event );
  void OnLogMessage( const std::string& msg );
  void OnMouseMove( wxMouseEvent& event );
  void OnMouseLeftDown( wxMouseEvent& event );
  void OnMouseLeftUp( wxMouseEvent& event );
  void OnMouseLeave( wxMouseEvent& event );
  void OnShow( wxShowEvent& event );

  wxSplitterWindow* mMainSplitter;
  wxSplitterWindow* mLeftSplitter;
  wxPanel* mMainPanel;
  wxPanel* mControlPanel;
  wxTextCtrl* mLogTextBox;

  // TODO lazy string as key
  std::unordered_map<std::string, wxNamedTextControl*> mParameterControls;
  using EventCallBack = std::function<void( wxCommandEvent& )>;
  std::unordered_map<std::string, std::pair<wxButton*, EventCallBack>> mButtons;

  GLCanvas::uptr mGLCanvas;
  rt::RayTracer::uptr mRayTracer;

  bool mCameraModeActive;
  math::vec2 mPreviousMouseScreenPosition;
};
