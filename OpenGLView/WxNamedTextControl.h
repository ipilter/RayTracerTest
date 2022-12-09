#pragma once

#include "WxMain.h"

class wxNamedTextControl : public wxWindow
{
public:
  wxNamedTextControl( wxWindow* parent
                      , wxWindowID id
                      , const std::string& name
                      , const std::string& defaultValue = "" );

  ~wxNamedTextControl();

  wxString GetValue() const;
  wxBoxSizer* GetSizer();

private:
  wxTextCtrl* mTextCtrl;
  wxStaticText* mName;
  wxBoxSizer* mSizer;
};
