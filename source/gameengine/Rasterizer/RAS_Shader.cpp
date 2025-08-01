/*
 * ***** BEGIN GPL LICENSE BLOCK *****
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
 *
 * ***** END GPL LICENSE BLOCK *****
 */

/** \file gameengine/Ketsji/RAS_Shader.cpp
 *  \ingroup bgerast
 */

#include "RAS_Shader.h"

#include "GPU_immediate.hh"

#include "CM_Message.h"

using namespace blender::gpu::shader;

RAS_Shader::RAS_Uniform::RAS_Uniform(int data_size)
    : m_loc(-1), m_count(1), m_dirty(true), m_type(UNI_NONE), m_transpose(0), m_dataLen(data_size)
{
#ifdef SORT_UNIFORMS
  m_data = (void *)MEM_mallocN(m_dataLen, "shader-uniform-alloc");
#endif
}

RAS_Shader::RAS_Uniform::~RAS_Uniform()
{
#ifdef SORT_UNIFORMS
  if (m_data) {
    MEM_freeN(m_data);
    m_data = nullptr;
  }
#endif
}

void RAS_Shader::RAS_Uniform::Apply(RAS_Shader *shader)
{
#ifdef SORT_UNIFORMS
  BLI_assert(m_type > UNI_NONE && m_type < UNI_MAX && m_data);

  if (!m_dirty) {
    return;
  }

  GPUShader *gpushader = shader->GetGPUShader();
  switch (m_type) {
    case UNI_FLOAT: {
      float *f = (float *)m_data;
      GPU_shader_uniform_float_ex(gpushader, m_loc, 1, m_count, (float *)f);
      break;
    }
    case UNI_INT: {
      int *f = (int *)m_data;
      GPU_shader_uniform_int_ex(gpushader, m_loc, 1, m_count, (int *)f);
      break;
    }
    case UNI_FLOAT2: {
      float *f = (float *)m_data;
      GPU_shader_uniform_float_ex(gpushader, m_loc, 2, m_count, (float *)f);
      break;
    }
    case UNI_FLOAT3: {
      float *f = (float *)m_data;
      GPU_shader_uniform_float_ex(gpushader, m_loc, 3, m_count, (float *)f);
      break;
    }
    case UNI_FLOAT4: {
      float *f = (float *)m_data;
      GPU_shader_uniform_float_ex(gpushader, m_loc, 4, m_count, (float *)f);
      break;
    }
    case UNI_INT2: {
      int *f = (int *)m_data;
      GPU_shader_uniform_int_ex(gpushader, m_loc, 2, m_count, (int *)f);
      break;
    }
    case UNI_INT3: {
      int *f = (int *)m_data;
      GPU_shader_uniform_int_ex(gpushader, m_loc, 3, m_count, (int *)f);
      break;
    }
    case UNI_INT4: {
      int *f = (int *)m_data;
      GPU_shader_uniform_int_ex(gpushader, m_loc, 4, m_count, (int *)f);
      break;
    }
    case UNI_MAT4: {
      float *f = (float *)m_data;
      GPU_shader_uniform_float_ex(gpushader, m_loc, 16, m_count, (float *)f);
      break;
    }
    case UNI_MAT3: {
      float *f = (float *)m_data;
      GPU_shader_uniform_float_ex(gpushader, m_loc, 9, m_count, (float *)f);
      break;
    }
  }
  m_dirty = false;
#endif
}

void RAS_Shader::RAS_Uniform::SetData(int location, int type, unsigned int count, bool transpose)
{
#ifdef SORT_UNIFORMS
  m_type = type;
  m_loc = location;
  m_count = count;
  m_dirty = true;
#endif
}

int RAS_Shader::RAS_Uniform::GetLocation()
{
  return m_loc;
}

void *RAS_Shader::RAS_Uniform::GetData()
{
  return m_data;
}

bool RAS_Shader::Ok() const
{
  return (m_shader && m_use);
}

RAS_Shader::RAS_Shader() : m_shader(nullptr), m_use(0), m_error(0), m_dirty(true)
{
  for (unsigned short i = 0; i < MAX_PROGRAM; ++i) {
    m_progs[i] = "";
  }
  m_constantUniforms = {};
  m_samplerUniforms = {};
}

RAS_Shader::~RAS_Shader()
{
  ClearUniforms();

  DeleteShader();
}

void RAS_Shader::ClearUniforms()
{
  for (RAS_Uniform *uni : m_uniforms) {
    delete uni;
  }
  m_uniforms.clear();

  for (RAS_DefUniform *uni : m_preDef) {
    delete uni;
  }
  m_preDef.clear();
}

RAS_Shader::RAS_Uniform *RAS_Shader::FindUniform(const int location)
{
#ifdef SORT_UNIFORMS
  for (RAS_Uniform *uni : m_uniforms) {
    if (uni->GetLocation() == location) {
      return uni;
    }
  }
#endif
  return nullptr;
}

void RAS_Shader::SetUniformfv(
    int location, int type, float *param, int size, unsigned int count, bool transpose)
{
#ifdef SORT_UNIFORMS
  RAS_Uniform *uni = FindUniform(location);

  if (uni) {
    memcpy(uni->GetData(), param, size);
    uni->SetData(location, type, count, transpose);
  }
  else {
    uni = new RAS_Uniform(size);
    memcpy(uni->GetData(), param, size);
    uni->SetData(location, type, count, transpose);
    m_uniforms.push_back(uni);
  }

  m_dirty = true;
#endif
}

void RAS_Shader::SetUniformiv(
    int location, int type, int *param, int size, unsigned int count, bool transpose)
{
#ifdef SORT_UNIFORMS
  RAS_Uniform *uni = FindUniform(location);

  if (uni) {
    memcpy(uni->GetData(), param, size);
    uni->SetData(location, type, count, transpose);
  }
  else {
    uni = new RAS_Uniform(size);
    memcpy(uni->GetData(), param, size);
    uni->SetData(location, type, count, transpose);
    m_uniforms.push_back(uni);
  }

  m_dirty = true;
#endif
}

void RAS_Shader::ApplyShader()
{
#ifdef SORT_UNIFORMS
  if (!m_dirty) {
    return;
  }

  for (unsigned int i = 0; i < m_uniforms.size(); i++) {
    m_uniforms[i]->Apply(this);
  }

  m_dirty = false;
#endif
}

void RAS_Shader::UnloadShader()
{
  //
}

void RAS_Shader::DeleteShader()
{
  if (m_shader) {
    GPU_shader_free(m_shader);
    m_shader = nullptr;
  }
}

void RAS_Shader::AppendUniformInfos(std::string type, std::string name)
{
  if (type == "float") {
    m_constantUniforms.push_back(UniformConstant({Type::float_t, name}));
  }
  else if (type == "int") {
    m_constantUniforms.push_back(UniformConstant({Type::int_t, name}));
  }
  else if (type == "vec2") {
    m_constantUniforms.push_back(UniformConstant({Type::float2_t, name}));
  }
  else if (type == "vec3") {
    m_constantUniforms.push_back(UniformConstant({Type::float3_t, name}));
  }
  else if (type == "vec4") {
    m_constantUniforms.push_back(UniformConstant({Type::float4_t, name}));
  }
  else if (type == "mat3") {
    m_constantUniforms.push_back(UniformConstant({Type::float3x3_t, name}));
  }
  else if (type == "mat4") {
    m_constantUniforms.push_back(UniformConstant({Type::float4x4_t, name}));
  }
  else if (type == "sampler2D") {
    if (m_samplerUniforms.size() > 7) {
      CM_Warning("RAS_Shader: Sampler index can't be > 7");
    }
    else {
      m_samplerUniforms.push_back({m_samplerUniforms.size(), name});
    }
  }
  else {
    CM_Warning("Invalid/unsupported uniform type: " << name);
  }
}

std::string RAS_Shader::GetParsedProgram(ProgramType type)
{
  std::string prog = m_progs[type];
  if (prog.empty()) {
    return prog;
  }

  const unsigned int pos = prog.find("#version");
  if (pos != -1) {
    CM_Warning("found redundant #version directive in shader program, directive ignored.");
    const unsigned int nline = prog.find("\n", pos);
    prog.erase(pos, nline - pos);
  }

  unsigned int uni_pos = prog.find("uniform");
  while (uni_pos != -1) {
    const unsigned int type_pos = prog.find(" ", uni_pos) + 1;
    const unsigned int name_pos = prog.find(" ", type_pos) + 1;
    const unsigned int end_namepos = prog.find(";", name_pos);
    std::string type = prog.substr(type_pos, (name_pos - 1) - type_pos);
    std::string name = prog.substr(name_pos, end_namepos - name_pos);
    AppendUniformInfos(type, name);

    prog.replace(uni_pos, 2, "//");

    const unsigned int endline_pos = prog.find("\n", end_namepos);

    uni_pos = prog.find("uniform", endline_pos);
  }

  prog.insert(0, "\n");

  return prog;
}

bool RAS_Shader::LinkProgram()
{
  std::string vert;
  std::string frag;
  std::string geom;

  vert = GetParsedProgram(VERTEX_PROGRAM);
  frag = GetParsedProgram(FRAGMENT_PROGRAM);
  geom = GetParsedProgram(GEOMETRY_PROGRAM);

  StageInterfaceInfo iface("s_Interface", "");
  iface.smooth(Type::float4_t, "bgl_TexCoord");

  ShaderCreateInfo info("s_Display");
  info.push_constant(Type::float_t, "bgl_RenderedTextureWidth");
  info.push_constant(Type::float_t, "bgl_RenderedTextureHeight");
  info.push_constant(Type::float2_t, "bgl_TextureCoordinateOffset", 9);
  for (std::pair<int, std::string> &sampler : m_samplerUniforms) {
    info.sampler(sampler.first, ImageType::Float2D, sampler.second);
  }
  info.sampler(8, ImageType::Float2D, "bgl_RenderedTexture");
  info.sampler(9, ImageType::Float2D, "bgl_DepthTexture");
  for (UniformConstant &constant : m_constantUniforms) {
    info.push_constant(constant.type, constant.name);
  }
  info.vertex_out(iface);
  info.fragment_out(0, Type::float4_t, "fragColor");
  info.vertex_source("draw_colormanagement_lib.glsl");
  info.fragment_source("draw_colormanagement_lib.glsl");
  info.vertex_source_generated = vert;
  info.fragment_source_generated = frag;

  if (m_error) {
    goto program_error;
  }

  if (m_progs[VERTEX_PROGRAM].empty() || m_progs[FRAGMENT_PROGRAM].empty()) {
    CM_Error("invalid GLSL sources.");
    return false;
  }

  m_shader = GPU_shader_create_from_info((GPUShaderCreateInfo *)&info);

  if (!m_shader) {
    goto program_error;
  }

  m_error = 0;
  return true;

program_error : {
  m_use = 0;
  m_error = 1;
  return false;
}
}

void RAS_Shader::ValidateProgram()
{
  char *log = GPU_shader_validate(m_shader);
  if (log) {
    CM_Debug("---- GLSL Validation ----\n" << log);
    MEM_freeN(log);
  }
}

bool RAS_Shader::GetError()
{
  return m_error;
}

GPUShader *RAS_Shader::GetGPUShader()
{
  return m_shader;
}

void RAS_Shader::SetSampler(int loc, int unit)
{
  //GPU_shader_uniform_int(m_shader, loc, unit);
}

void RAS_Shader::SetProg(bool enable)
{
  if (m_shader && enable) {
    immBindShader(m_shader);
  }
  else {
    immUnbindProgram();
  }
}

void RAS_Shader::SetEnabled(bool enabled)
{
  m_use = enabled;
}

bool RAS_Shader::GetEnabled() const
{
  return m_use;
}

void RAS_Shader::Update(RAS_Rasterizer *rasty, const MT_Matrix4x4 model)
{
  if (!Ok() || m_preDef.empty()) {
    return;
  }

  const MT_Matrix4x4 &view = rasty->GetViewMatrix();

  for (RAS_DefUniform *uni : m_preDef) {
    if (uni->m_loc == -1) {
      continue;
    }

    switch (uni->m_type) {
      case MODELMATRIX: {
        SetUniform(uni->m_loc, model);
        break;
      }
      case MODELMATRIX_TRANSPOSE: {
        SetUniform(uni->m_loc, model, true);
        break;
      }
      case MODELMATRIX_INVERSE: {
        SetUniform(uni->m_loc, model.inverse());
        break;
      }
      case MODELMATRIX_INVERSETRANSPOSE: {
        SetUniform(uni->m_loc, model.inverse(), true);
        break;
      }
      case MODELVIEWMATRIX: {
        SetUniform(uni->m_loc, view * model);
        break;
      }
      case MODELVIEWMATRIX_TRANSPOSE: {
        MT_Matrix4x4 mat(view * model);
        SetUniform(uni->m_loc, mat, true);
        break;
      }
      case MODELVIEWMATRIX_INVERSE: {
        MT_Matrix4x4 mat(view * model);
        SetUniform(uni->m_loc, mat.inverse());
        break;
      }
      case MODELVIEWMATRIX_INVERSETRANSPOSE: {
        MT_Matrix4x4 mat(view * model);
        SetUniform(uni->m_loc, mat.inverse(), true);
        break;
      }
      case CAM_POS: {
        MT_Vector3 pos(rasty->GetCameraPosition());
        SetUniform(uni->m_loc, pos);
        break;
      }
      case VIEWMATRIX: {
        SetUniform(uni->m_loc, view);
        break;
      }
      case VIEWMATRIX_TRANSPOSE: {
        SetUniform(uni->m_loc, view, true);
        break;
      }
      case VIEWMATRIX_INVERSE: {
        SetUniform(uni->m_loc, view.inverse());
        break;
      }
      case VIEWMATRIX_INVERSETRANSPOSE: {
        SetUniform(uni->m_loc, view.inverse(), true);
        break;
      }
      case CONSTANT_TIMER: {
        SetUniform(uni->m_loc, (float)rasty->GetTime());
        break;
      }
      case EYE: {
        SetUniform(uni->m_loc,
                   (rasty->GetEye() == RAS_Rasterizer::RAS_STEREO_LEFTEYE) ? 0.0f : 0.5f);
      }
      default:
        break;
    }
  }
}

int RAS_Shader::GetAttribLocation(const std::string &name)
{
  return GPU_shader_get_attribute(m_shader, name.c_str());
}

//void RAS_Shader::BindAttributes(const std::unordered_map<int, std::string> &attrs)
//{
//  const unsigned short len = attrs.size();
//  int *locations = (int *)BLI_array_alloca(locations, len);
//  const char **names = (const char **)BLI_array_alloca(names, len);
//
//  unsigned short i = 0;
//  for (const std::pair<int, std::string> &pair : attrs) {
//    locations[i] = pair.first;
//    names[i] = pair.second.c_str();
//    ++i;
//  }
//
//  GPU_shader_bind_attributes(m_shader, locations, (const char **)names, len);
//}

int RAS_Shader::GetUniformLocation(const std::string &name, bool debug)
{
  BLI_assert(m_shader != nullptr);
  int location = GPU_shader_get_uniform(m_shader, name.c_str());

  if (location == -1 && debug) {
    CM_Error("invalid uniform value: " << name << ".");
  }

  return location;
}

void RAS_Shader::SetUniform(int uniform, const MT_Vector2 &vec)
{
  float value[2];
  vec.getValue(value);
  GPU_shader_uniform_float_ex(m_shader, uniform, 2, 1, value);
}

void RAS_Shader::SetUniform(int uniform, const MT_Vector3 &vec)
{
  float value[3];
  vec.getValue(value);
  GPU_shader_uniform_float_ex(m_shader, uniform, 3, 1, value);
}

void RAS_Shader::SetUniform(int uniform, const MT_Vector4 &vec)
{
  float value[4];
  vec.getValue(value);
  GPU_shader_uniform_float_ex(m_shader, uniform, 4, 1, value);
}

void RAS_Shader::SetUniform(int uniform, const unsigned int &val)
{
  GPU_shader_uniform_int_ex(m_shader, uniform, 1, 1, (int *)&val);
}

void RAS_Shader::SetUniform(int uniform, const int val)
{
  GPU_shader_uniform_int_ex(m_shader, uniform, 1, 1, (int *)&val);
}

void RAS_Shader::SetUniform(int uniform, const float &val)
{
  GPU_shader_uniform_float_ex(m_shader, uniform, 1, 1, (float *)&val);
}

void RAS_Shader::SetUniform(int uniform, const MT_Matrix4x4 &vec, bool transpose)
{
  float value[16];
  // note: getValue gives back column major as needed by OpenGL
  vec.getValue(value);
  GPU_shader_uniform_float_ex(m_shader, uniform, 16, 1, value);
}

void RAS_Shader::SetUniform(int uniform, const MT_Matrix3x3 &vec, bool transpose)
{
  float value[9];
  value[0] = (float)vec[0][0];
  value[1] = (float)vec[1][0];
  value[2] = (float)vec[2][0];
  value[3] = (float)vec[0][1];
  value[4] = (float)vec[1][1];
  value[5] = (float)vec[2][1];
  value[6] = (float)vec[0][2];
  value[7] = (float)vec[1][2];
  value[8] = (float)vec[2][2];
  GPU_shader_uniform_float_ex(m_shader, uniform, 9, 1, value);
}

void RAS_Shader::SetUniform(int uniform, const float *val, int len)
{
  if (len >= 2 && len <= 4) {
    GPU_shader_uniform_float_ex(m_shader, uniform, len, 1, (float *)val);
  }
  else {
    BLI_assert(0);
  }
}

void RAS_Shader::SetUniform(int uniform, const int *val, int len)
{
  if (len >= 2 && len <= 4) {
    GPU_shader_uniform_int_ex(m_shader, uniform, len, 1, (int *)val);
  }
  else {
    BLI_assert(0);
  }
}
