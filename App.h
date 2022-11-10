#pragma once

#include "WxMain.h"
#include "Math.h"

class MainFrame;

class App : public wxApp
{
public:
  virtual bool OnInit();

private:
  virtual int OnExit();
  void OnKey( wxKeyEvent& event );

  virtual void OnInitCmdLine(wxCmdLineParser& parser);
  virtual bool OnCmdLineParsed(wxCmdLineParser& parser);

private:
  MainFrame* mMainFrame;
  math::uvec2 mTextureSize;
};
