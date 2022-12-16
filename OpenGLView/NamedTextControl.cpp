#include "NamedTextControl.h"
#include "Common/HostUtils.h"
#include "Common/Math.h"

NamedTextControl::NamedTextControl( wxWindow* parent
                                    , wxWindowID id
                                    , const std::string& name
                                    , const std::string& defaultValue
                                    , const float delta
                                    , const float smallDelta
                                    , const float bigDelta
                                    , const float min
                                    , const float max )
  : wxWindow( parent, id )
  , mTextCtrl( new wxTextCtrl( this, wxID_ANY ) )
  , mName( new wxStaticText( this, wxID_ANY, name ) )
  , mSizer( new wxBoxSizer( wxHORIZONTAL ) )
  , mDelta( delta )
  , mSmallDelta( smallDelta )
  , mBigDelta( bigDelta )
  , mMin( min )
  , mMax( max )
{
  mSizer->Add( mName );
  mSizer->AddStretchSpacer();
  mSizer->Add( mTextCtrl );
  SetSizer( mSizer );
  mTextCtrl->SetValue( defaultValue );

  Bind( wxEVT_MOUSEWHEEL, &NamedTextControl::OnMouseWheel, this );

  Layout();
}

NamedTextControl::~NamedTextControl()
{}

wxString NamedTextControl::GetValue() const
{
  return mTextCtrl->GetValue();
}

void NamedTextControl::SetOnMouseWheelCallback( const MessageCallBack& callback )
{
  mOnMouseWheelCallback = callback;
}

void NamedTextControl::OnMouseWheel( wxMouseEvent& event )
{
  const bool useBigDelta( wxGetKeyState( WXK_SHIFT ) );
  const bool useSmallDelta( wxGetKeyState( WXK_CONTROL ) );
  const float deltaValue = useBigDelta ? mBigDelta : useSmallDelta ? mSmallDelta : mDelta;
  const float delta( event.GetWheelRotation() < 0 ? -deltaValue : deltaValue );

  const std::string oldValue( GetValue().ToStdString() );
  const std::string newValue( util::ToString( glm::clamp( util::FromString<float>( oldValue ) + delta, mMin, mMax ) ) );
  if ( newValue == oldValue )
  {
    return;
  }

  mTextCtrl->SetValue( newValue );
  if ( mOnMouseWheelCallback != nullptr )
  {
    mOnMouseWheelCallback();
  }
}
