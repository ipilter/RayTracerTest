#pragma once

#include <gl/glew.h> // before any gl related include
#include "WxMain.h"
#include <wx/glcanvas.h>
#include "Common\Math.h"
#include "Common\Sptr.h"

class GLCanvas : public wxGLCanvas, public virtual ISptr<GLCanvas>
{
public:
  GLCanvas( const math::uvec2& imageSize
          , wxWindow* parent
          , wxWindowID id = wxID_ANY
          , const int* attribList = 0
          , const wxPoint& pos = wxDefaultPosition
          , const wxSize& size = wxDefaultSize
          , long style = 0L
          , const wxString& name = L"GLCanvas"
          , const wxPalette& palette = wxNullPalette );

  const math::uvec2& ImageSize() const;

private:
  // opengl context
  std::unique_ptr<wxGLContext> mContext;

  math::uvec2 mImageSize;
};
