#pragma once

#include <sstream>
#include <fstream>
#include <stdexcept>

namespace util
{

template<class T>
inline T FromString( const std::string& str )
{
  T t( 0 );
  if ( !( std::istringstream( str ) >> t ) )
  {
    throw std::runtime_error( std::string( "cannot parse " ) + str + " as number" );
  }
  return t;
}

template<class T>
inline std::string ToString( const T& t )
{
  std::stringstream ss;
  ss << t;
  return ss.str();
}

inline std::string ReadTextFile( const std::string& path )
{
  std::ifstream is( path );
  if ( !is.is_open() )
  {
    throw std::runtime_error( std::string( "cannot open file: " ) + path );
  }
  return std::string( ( std::istreambuf_iterator<char>( is ) ), std::istreambuf_iterator<char>() );
}

template<typename T>
inline std::ofstream& Write_t( std::ofstream& stream, const T& t )
{
  stream.write( reinterpret_cast<const char*>( &t ), sizeof( T ) );
  return stream;
}

template<>
inline std::ofstream& Write_t( std::ofstream& stream, const std::string& str )
{
  Write_t( stream, str.length() );
  stream.write( str.c_str(), str.length() );
  return stream;
}

template<typename T>
inline std::ifstream& Read_t( std::ifstream& stream, T& t )
{
  stream.read( reinterpret_cast<char*> ( &t ), sizeof( T ) );
  return stream;
}

template<>
inline std::ifstream& Read_t( std::ifstream& stream, std::string& str )
{
  size_t count( 0 );
  Read_t( stream, count );
  str.reserve( count );

  std::istreambuf_iterator<char> chars( stream );
  std::copy_n( chars, count, std::back_inserter<std::string>( str ) );

  char dummy( 0 );
  Read_t( stream, dummy );
  return stream;
}

}
