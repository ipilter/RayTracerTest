#pragma once

#include <sstream>
#include <fstream>
#include <stdexcept>
#include <random>

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

// TODO T can be float or double only!
template<class T>
inline T Random( T min = 0, T max = 1 )
{
  static std::random_device rd;
  static std::mt19937 gen( rd() );
  static std::uniform_real_distribution<T> dist( min, max );

  return dist( gen );
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

template<class T>
inline T Clamp( const T a, const T b, const T v )
{
  return v < a ? a : v > b ? b : v;
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

inline uint8_t Component( const uint32_t& color, const uint32_t& idx )
{
  static const uint32_t params[][2]{ {0x000000FF, 0}, {0x0000FF00, 8}, {0x00FF0000, 16}, {0xFF000000, 24} };   // RGBA
  return static_cast<uint8_t>( ( color & params[idx][0] ) >> params[idx][1] );
}

inline uint32_t Color( const uint8_t r = 0, const uint8_t g = 0, const uint8_t b = 0, const uint8_t a = 255 )
{
  return ( b << 0 ) | ( g << 8 ) | ( r << 16 ) | ( a << 24 );   // BGRA
}

}
