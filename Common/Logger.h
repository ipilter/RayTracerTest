#pragma once

#include <memory>
#include <fstream>

#include "Math.h"

namespace logger
{

class Logger
{
public:
  ~Logger();

  template<class T>
  Logger& operator << ( const T& t )
  {
    logStream << t;
    logStream.flush();
    return *this;
  }

  static Logger& Instance();

private:
  Logger();

  std::ofstream logStream;

  Logger( const Logger& rhs ) = delete;
  Logger( Logger&& rhs ) noexcept = delete;
  Logger& operator=( const Logger& rhs ) = delete;
  Logger& operator=( Logger&& rhs ) noexcept = delete;
};

}
