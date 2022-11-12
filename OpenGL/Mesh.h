#pragma once

#include <vector>

#include "Common\Sptr.h"
#include "Common\Math.h"

namespace gl
{

class Mesh : public virtual ISptr<Mesh>
{
public:
  // vertices format: vtx.x, vtx.y, tx.x, tx.y, ...
  // indices format: triangle vertex indices
  Mesh( const std::vector<float>& vertices, const std::vector<uint32_t>& indices );
  ~Mesh();

  void Draw() const;

private:
  uint32_t mVbo;
  uint32_t mIbo;
  uint32_t mVao;
  uint32_t mIndexCount;
};

}
