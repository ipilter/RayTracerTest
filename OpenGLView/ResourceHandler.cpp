#include "ResourceHandler.h"

#include <Windows.h>

HMODULE GetCurrentModule()
{
  HMODULE hModule = NULL;
  GetModuleHandleEx( GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS, (LPCTSTR)GetCurrentModule, &hModule );
  return hModule;
}

std::string LoadStringResource( HMODULE hModule, const int id, const int type )
{
  HRSRC hResf = FindResource( hModule, MAKEINTRESOURCE( id ), MAKEINTRESOURCE( type ) );
  HGLOBAL hDataf = LoadResource(hModule, hResf ) ;
  return std::string( static_cast<char*>( LockResource( hDataf ) ), static_cast<size_t>( SizeofResource( hModule, hResf ) ) );
}

std::string LoadStringResource( const int id, const int type )
{
  return LoadStringResource( GetCurrentModule(), id, type );
}
