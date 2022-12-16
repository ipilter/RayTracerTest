#pragma once

#include "WxMain.h"

// TODO currently the value is stored as a string to be type independent <store it in std containser> 
// but would be great to have typed (int, uint, float, etc) for cleaner interface
class NamedTextControl : public wxWindow
{
public:
  using MessageCallBack = std::function<void()>; // called on OnMouseWheel if bound

public:
  NamedTextControl( wxWindow* parent
                      , wxWindowID id
                      , const std::string& name
                      , const std::string& defaultValue
                      , const float delta
                      , const float smallDelta
                      , const float bigDelta
                      , const float min
                      , const float max );

  ~NamedTextControl();

  wxString GetValue() const;
  void SetOnMouseWheelCallback( const MessageCallBack& callback );

private:
  void OnMouseWheel( wxMouseEvent& event );
  MessageCallBack mOnMouseWheelCallback;

  wxTextCtrl* mTextCtrl;
  wxStaticText* mName;
  wxBoxSizer* mSizer;
  float mDelta;
  float mSmallDelta;
  float mBigDelta;
  float mMin;
  float mMax;
};
