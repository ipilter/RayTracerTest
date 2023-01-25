#pragma once

#include <functional>

#include "Common\Color.h"

namespace rt
{
using CallBackFunction = std::function<void( rt::ColorPtr deviceImageBuffer, const std::size_t size )>;
}
