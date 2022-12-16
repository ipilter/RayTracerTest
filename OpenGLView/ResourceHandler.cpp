#include "ResourceHandler.h"

#include <Windows.h>

namespace resource
{

HMODULE GetCurrentModule()
{
  HMODULE hModule = NULL;
  GetModuleHandleEx( GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS, (LPCTSTR)GetCurrentModule, &hModule );
  return hModule;
}

std::string GetStringResource( HMODULE hModule, const int id, const int type )
{
  // TODO error handling
  HRSRC hResf = FindResource( hModule, MAKEINTRESOURCE( id ), MAKEINTRESOURCE( type ) );
  HGLOBAL hDataf = LoadResource(hModule, hResf ) ;
  return std::string( static_cast<char*>( LockResource( hDataf ) ), static_cast<size_t>( SizeofResource( hModule, hResf ) ) );
}

std::string LoadString( const int id, const int type )
{
  return GetStringResource( GetCurrentModule(), id, type );
}

}
