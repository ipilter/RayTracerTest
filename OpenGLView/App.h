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

private:
  MainFrame::uptr mMainFrame;
};
