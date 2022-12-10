#include "WxNamedTextControl.h"

wxNamedTextControl::wxNamedTextControl( wxWindow* parent
                                        , wxWindowID id
                                        , const std::string& name
                                        , const std::string& defaultValue )
  : wxWindow( parent, id )
  , mTextCtrl( new wxTextCtrl( this, wxID_ANY ) )
  , mName( new wxStaticText( this, wxID_ANY, name ) )
  , mSizer( new wxBoxSizer( wxHORIZONTAL ) )
{
  mSizer->Add( mName );
  mSizer->AddStretchSpacer();
  mSizer->Add( mTextCtrl );
  SetSizer( mSizer );
  mTextCtrl->SetValue( defaultValue );
  Layout();
}

wxNamedTextControl::~wxNamedTextControl()
{}

wxString wxNamedTextControl::GetValue() const
{
  return mTextCtrl->GetValue();
}

wxBoxSizer* wxNamedTextControl::GetSizer()
{
  return mSizer;
}
