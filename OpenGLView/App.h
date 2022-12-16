#pragma once

#include "WxMain.h"
#include "Common\Math.h"

#include "MainFrame.h"

class App : public wxApp
{
public:
  App();
  ~App();

  virtual bool OnInit();
  virtual int OnExit();

private:
  virtual void OnInitCmdLine(wxCmdLineParser& parser);
  virtual bool OnCmdLineParsed(wxCmdLineParser& parser);

  MainFrame::uptr mMainFrame;
  math::uvec2 mTextureSize;
  uint32_t mSampleCount;
  float mFov;
  float mFocalLength;
  float mAperture;
};
