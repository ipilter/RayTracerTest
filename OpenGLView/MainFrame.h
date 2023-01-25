#pragma once

#include <wx\splitter.h>

#include "GLCanvas.h"
#include "NamedTextControl.h"
#include "RayTracer\RayTracer.h"
#include "RayTracer\RaytracerCallback.h"

class MainFrame : public wxFrame, public virtual ISptr<MainFrame>
{
public:
  MainFrame( const math::uvec2& imageSize
             , const uint32_t sampleCount
             , const uint32_t iterationCount
             , const uint32_t updateInterval
             , const math::vec3& cameraPosition
             , const math::vec2& cameraAngles
             , const float fov
             , const float focalLength
             , const float aperture
             , const math::vec2& anglesPerAxes
             , wxWindow* parent
             , std::wstring title
             , const wxPoint& pos
             , const wxSize& size );

  virtual ~MainFrame();

  void TracerUpdateCallback( rt::ColorPtr deviceImageBuffer, const std::size_t size );
  void TracerFinishedCallback( rt::ColorPtr deviceImageBuffer, const std::size_t size );

private:
  void InitializeUIElements();
  void RequestTrace();
  
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
  void OnTracerUpdate();
  void OnTracerFinished();

  wxSplitterWindow* mMainSplitter;
  wxSplitterWindow* mLeftSplitter;
  wxPanel* mMainPanel;
  wxPanel* mControlPanel;
  wxTextCtrl* mLogTextBox;

  // TODO lazy string as key
  std::vector<NamedTextControl*> mParameterControls;
  using EventCallBack = std::function<void( wxCommandEvent& )>;
  std::vector<std::pair<std::string, std::pair<wxButton*, EventCallBack>>> mButtons;

  GLCanvas::uptr mGLCanvas;
  rt::RayTracer::uptr mRayTracer;
  
  // TODO reference to raytracer's imagebuffer
  // wrap these into a single variable
  rt::ColorPtr mDeviceImageBuffer;
  std::size_t mSize;

  bool mCameraModeActive;
  math::vec2 mPreviousMouseScreenPosition;
  math::vec2 mAnglePerAxes;
};
