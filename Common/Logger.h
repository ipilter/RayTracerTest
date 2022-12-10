#pragma once

#include <memory>
#include <functional>
#include <fstream>

#include "Math.h"
#include "HostUtils.h"

namespace logger
{

class Logger
{
public:
  using MessageCallBack = std::function<void( const std::string& )>;

public:
  ~Logger();

  template<class T>
  Logger& operator << ( const T& t )
  {
    if ( mMessageCallback != nullptr )
    {
      // TODO use std::endl like line closing
      const std::string tStr( util::ToString( t ) );
      mLineCache << tStr;
      if ( !tStr.empty() && tStr[tStr.size() - 1] == '\n' )
      {
        mMessageCallback( mLineCache.str() );
        mLineCache = std::stringstream(); // TODO do it better
      }
    }

    logStream << t;
    logStream.flush();
    return *this;
  }

  void SetMessageCallback( MessageCallBack callback = nullptr );

  static Logger& Instance();

private:
  Logger();

  std::ofstream logStream;
  
  MessageCallBack mMessageCallback;

  Logger( const Logger& rhs ) = delete;
  Logger( Logger&& rhs ) noexcept = delete;
  Logger& operator=( const Logger& rhs ) = delete;
  Logger& operator=( Logger&& rhs ) noexcept = delete;

  std::stringstream mLineCache;
};

}
