#pragma once

#include "Math.h"

namespace math
{

// TODO T can be float or double only!
template<class T>
inline T Random( T min = 0, T max = 1 )
{
  static std::random_device rd;
  static std::mt19937 gen( rd() );
  static std::uniform_real_distribution<T> dist( min, max );

  return dist( gen );
}

}
