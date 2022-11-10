#pragma once
#include <memory>

template<class T>
class ISptr
{
public:
  using sptr = std::shared_ptr<T>;
  using uptr = std::unique_ptr<T>;
};
