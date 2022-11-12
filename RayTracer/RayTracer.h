#pragma once

#include "Common\Sptr.h"
#include "RenderData.h"

namespace rt
{

class RayTracer : public ISptr<RayTracer>
{
public:
  RayTracer();
  void Trace( rt::RenderData& renderData );

private:
};

}
