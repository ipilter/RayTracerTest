#pragma once

#include <gl/glew.h>

#include "Common\Sptr.h"
#include "Common\Math.h"
#include "Common\Color.h"

namespace gl
{

class Texture : public virtual ISptr<Texture>
{
public:
  Texture( const math::uvec2& size, const uint32_t wrap = GL_CLAMP );
  ~Texture();

  void Bind();
  void Unbind();

  void BindTextureUnit( const uint32_t unitId );
  void UnbindTextureUnit();

  void CreateFromArray( const rt::color_t* array );
  void CreateFromPBO();
  void UpdateFromPBO();
  void UpdateFromPBO( uint32_t regionPosX, uint32_t regionPosY, uint32_t regionWidth, uint32_t regionHeight );

  const math::uvec2& Size() const;
  uint32_t ChannelCount() const;
  uint32_t BytesPerChannel() const;

private:
  uint32_t mId;
  math::uvec2 mSize;
  uint32_t mChannelCount;
};

}
