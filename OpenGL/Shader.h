#pragma once

#include "..\Sptr.h"
#include "..\Math.h"

namespace gl
{

class Shader : public virtual ISptr<Shader>
{
public:
  Shader( const std::string& vertexShaderSrc, const std::string& fragentShaderSrc );
  ~Shader();

  void Bind();
  void UnBind();

  void UpdateUniform( const std::string& name, const math::mat4& matrix );
  // TODO the rest...

private:
  void CreateShaderProgram( const std::string& vertexShaderSrc, const std::string& fragentShaderSrc );

  uint32_t CreateShader( uint32_t kind, const std::string& src );

  uint32_t mVertexShader = -1;
  uint32_t mFragmentxShader = -1;
  uint32_t mShaderProgram = -1;
};

}