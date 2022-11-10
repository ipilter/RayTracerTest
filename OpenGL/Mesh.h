#pragma once

#include <vector>

#include "..\Sptr.h"
#include "..\Math.h"

namespace gl
{

class Mesh : public virtual ISptr<Mesh>
{
public:
  // vertices format: vtx.x, vtx.y, tx.x, tx.y, ...
  // indices format: triangle vertex indices
  Mesh( const std::vector<float>& vertices, const std::vector<uint32_t>& indices );
  ~Mesh();

  void Draw();

private:
  uint32_t mVbo = -1;
  uint32_t mIbo = -1;
  uint32_t mVao = -1;
  uint32_t mIndexCount = -1;
};

}
