#pragma once

#include <wx\splitter.h>

#include "GLCanvas.h"
#include "NamedTextControl.h"
#include "RayTracer\RayTracer.h"
#include "RayTracer\RaytracerCallback.h"

#include "Common\Timer.h"

class MainFrame : public wxFrame, public virtual ISptr<MainFrame>
{
  using ControlInitialParameters = std::tuple<std::string, uint32_t, float, float, float, float, float>;
  using ControlInitialParametersList = std::list<ControlInitialParameters>;

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
  void InitializeUIElements( const ControlInitialParametersList& parameters );
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
  void OnMouseWheel( wxMouseEvent& event );

  void OnMouseRightDown( wxMouseEvent& event );
  void OnMouseRightUp( wxMouseEvent& event );
  void OnMouseMiddleDown( wxMouseEvent& event );
  void OnMouseMiddleUp( wxMouseEvent& event );

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

  // Tracer control
  // used for interacting the raytracer camera
  bool mIsTracerCameraMode;
  // moving the mouse cursor across the screen results this amount of rotation of the tracer camera.
  // (smaller the value, slower the camera movement. 
  math::vec2 mAnglePerAxes;
  // filter out update request on mouse move. update for every pixel movement is too much
  Timer mTimer;
  double mLastTime;

  // OpenGL view control
  // used for interacting the openg view camera
  bool mIsViewCameraMode;
  // mouse position in GLCanvas coordinate space
  math::vec2 mPreviousMouseScreenPosition;
};
