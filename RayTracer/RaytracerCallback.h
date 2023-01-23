#pragma once

#include <functional>

#include "Common\Color.h"

namespace rt
{
using CallBackFunction = std::function<void( rt::ColorConstPtr deviceImageBuffer, const std::size_t size )>;
}
