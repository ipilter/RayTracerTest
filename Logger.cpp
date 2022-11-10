#include "Logger.h"

namespace logger
{

Logger& Logger::Instance()
{
  static Logger instance; // static initialization is thread-safe in C++11. !! after VS2015 !!
                          //volatile int dummy{}; // max optimization can remove this mehtod?
  return instance;
}

Logger::~Logger()
{}

Logger::Logger()
{
  logStream.open( "e:\\default.log" );
}

}
