/* SPDX-FileCopyrightText: 2023 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup gpu
 */

#pragma once

#include "BLI_map.hh"
#include "BLI_span.hh"
#include "BLI_string_ref.hh"

#include "GPU_shader.hh"
#include "GPU_worker.hh"
#include "gpu_shader_create_info.hh"
#include "gpu_shader_interface.hh"

#include <deque>
#include <string>

namespace blender::gpu {

class GPULogParser;
class Context;

/* Set to 1 to log the full source of shaders that fail to compile. */
#define DEBUG_LOG_SHADER_SRC_ON_ERROR 0

/**
 * Compilation is done on a list of GLSL sources. This list contains placeholders that should be
 * provided by the backend shader. These defines contains the locations where the backend can patch
 * the sources.
 */
#define SOURCES_INDEX_VERSION 0
#define SOURCES_INDEX_SPECIALIZATION_CONSTANTS 1

/**
 * Implementation of shader compilation and uniforms handling.
 * Base class which is then specialized for each implementation (GL, VK, ...).
 */
class Shader {
 public:
  /** Uniform & attribute locations for shader. */
  ShaderInterface *interface = nullptr;
  /** Bit-set indicating the frame-buffer color attachments that this shader writes to. */
  uint16_t fragment_output_bits = 0;

  /**
   * Specialization constants as a Struct-of-Arrays. Allow simpler comparison and reset.
   * The backend is free to implement their support as they see fit.
   */
  struct Constants {
    using Value = shader::SpecializationConstant::Value;
    Vector<gpu::shader::Type> types;
    /* Current values set by `GPU_shader_constant_*()` call. The backend can choose to interpret
     * that however it wants (i.e: bind another shader instead). */
    Vector<Value> values;

    /**
     * OpenGL needs to know if a different program needs to be attached when constants are
     * changed. Vulkan and Metal uses pipelines and don't have this issue. Attribute can be
     * removed after the OpenGL backend has been phased out.
     */
    bool is_dirty;
  } constants;

  /* WORKAROUND: True if this shader is a polyline shader and needs an appropriate setup to render.
   * Eventually, in the future, we should modify the user code instead of relying on such hacks. */
  bool is_polyline = false;

 protected:
  /** For debugging purpose. */
  char name[64];

  /* Parent shader can be used for shaders which are derived from the same source material.
   * The child shader can pull information from its parent to prepare additional resources
   * such as PSOs upfront. This enables asynchronous PSO compilation which mitigates stuttering
   * when updating new materials. */
  Shader *parent_shader_ = nullptr;

 public:
  Shader(const char *name);
  virtual ~Shader();

  /* `is_batch_compilation` is true when the shader is being compiled as part of a
   * `GPU_shader_batch`. Backends that use the `ShaderCompilerGeneric` can ignore it. */
  virtual void init(const shader::ShaderCreateInfo &info, bool is_batch_compilation) = 0;

  virtual void vertex_shader_from_glsl(MutableSpan<StringRefNull> sources) = 0;
  virtual void geometry_shader_from_glsl(MutableSpan<StringRefNull> sources) = 0;
  virtual void fragment_shader_from_glsl(MutableSpan<StringRefNull> sources) = 0;
  virtual void compute_shader_from_glsl(MutableSpan<StringRefNull> sources) = 0;
  virtual bool finalize(const shader::ShaderCreateInfo *info = nullptr) = 0;
  /* Pre-warms PSOs using parent shader's cached PSO descriptors. Limit specifies maximum PSOs to
   * warm. If -1, compiles all PSO permutations in parent shader.
   *
   * See `GPU_shader_warm_cache(..)` in `GPU_shader.hh` for more information. */
  virtual void warm_cache(int limit) = 0;

  virtual void bind() = 0;
  virtual void unbind() = 0;

  virtual void uniform_float(int location, int comp_len, int array_size, const float *data) = 0;
  virtual void uniform_int(int location, int comp_len, int array_size, const int *data) = 0;

  /* Add specialization constant declarations to shader instance. */
  void specialization_constants_init(const shader::ShaderCreateInfo &info);

  std::string defines_declare(const shader::ShaderCreateInfo &info) const;
  virtual std::string resources_declare(const shader::ShaderCreateInfo &info) const = 0;
  virtual std::string vertex_interface_declare(const shader::ShaderCreateInfo &info) const = 0;
  virtual std::string fragment_interface_declare(const shader::ShaderCreateInfo &info) const = 0;
  virtual std::string geometry_interface_declare(const shader::ShaderCreateInfo &info) const = 0;
  virtual std::string geometry_layout_declare(const shader::ShaderCreateInfo &info) const = 0;
  virtual std::string compute_layout_declare(const shader::ShaderCreateInfo &info) const = 0;

  StringRefNull name_get() const
  {
    return name;
  }

  void parent_set(Shader *parent)
  {
    parent_shader_ = parent;
  }

  Shader *parent_get() const
  {
    return parent_shader_;
  }

  static bool srgb_uniform_dirty_get();
  static void set_srgb_uniform(GPUShader *shader);
  static void set_framebuffer_srgb_target(int use_srgb_to_linear);

  /* UPBGE */
  virtual char *shader_validate() = 0;
  //virtual void shader_bind_attributes(int *locations, const char **names, int len) = 0;
  /* GPU_shader_get_uniform doesn't handle array uniforms e.g: uniform vec2
     bgl_TextureCoordinateOffset[9]; */
  /*********/

 protected:
  void print_log(Span<StringRefNull> sources,
                 const char *log,
                 const char *stage,
                 bool error,
                 GPULogParser *parser);
};

/* Syntactic sugar. */
static inline GPUShader *wrap(Shader *vert)
{
  return reinterpret_cast<GPUShader *>(vert);
}
static inline Shader *unwrap(GPUShader *vert)
{
  return reinterpret_cast<Shader *>(vert);
}
static inline const Shader *unwrap(const GPUShader *vert)
{
  return reinterpret_cast<const Shader *>(vert);
}

class ShaderCompiler {
 protected:
  struct Sources {
    std::string vert;
    std::string geom;
    std::string frag;
    std::string comp;
  };

 public:
  virtual ~ShaderCompiler() = default;

  Shader *compile(const shader::ShaderCreateInfo &info, bool is_batch_compilation);

  virtual BatchHandle batch_compile(Span<const shader::ShaderCreateInfo *> &infos) = 0;
  virtual bool batch_is_ready(BatchHandle handle) = 0;
  virtual Vector<Shader *> batch_finalize(BatchHandle &handle) = 0;

  virtual SpecializationBatchHandle precompile_specializations(
      Span<ShaderSpecialization> /*specializations*/)
  {
    /* No-op. */
    return 0;
  };

  virtual bool specialization_batch_is_ready(SpecializationBatchHandle &handle)
  {
    handle = 0;
    return true;
  };
};

/* Generic implementation used as fallback. */
class ShaderCompilerGeneric : public ShaderCompiler {
 private:
  struct Batch {
    Vector<Shader *> shaders;
    Vector<const shader::ShaderCreateInfo *> infos;
    std::atomic_bool is_ready = false;
  };
  BatchHandle next_batch_handle_ = 1;
  Map<BatchHandle, std::unique_ptr<Batch>> batches_;
  std::mutex mutex_;

  std::deque<Batch *> compilation_queue_;
  std::unique_ptr<GPUWorker> compilation_thread_;

  void run_thread();

 public:
  ShaderCompilerGeneric();
  ~ShaderCompilerGeneric() override;

  BatchHandle batch_compile(Span<const shader::ShaderCreateInfo *> &infos) override;
  bool batch_is_ready(BatchHandle handle) override;
  Vector<Shader *> batch_finalize(BatchHandle &handle) override;
};

enum class Severity {
  Unknown,
  Warning,
  Error,
  Note,
};

struct LogCursor {
  int source = -1;
  int row = -1;
  int column = -1;
  StringRef file_name_and_error_line = {};
};

struct GPULogItem {
  LogCursor cursor;
  Severity severity = Severity::Unknown;
};

class GPULogParser {
 public:
  virtual const char *parse_line(const char *source_combined,
                                 const char *log_line,
                                 GPULogItem &log_item) = 0;

 protected:
  const char *skip_severity(const char *log_line,
                            GPULogItem &log_item,
                            const char *error_msg,
                            const char *warning_msg,
                            const char *note_msg) const;
  const char *skip_separators(const char *log_line, const StringRef separators) const;
  const char *skip_until(const char *log_line, char stop_char) const;
  bool at_number(const char *log_line) const;
  bool at_any(const char *log_line, const StringRef chars) const;
  int parse_number(const char *log_line, const char **r_new_position) const;

  MEM_CXX_CLASS_ALLOC_FUNCS("GPULogParser");
};

void printf_begin(Context *ctx);
void printf_end(Context *ctx);

}  // namespace blender::gpu

/* XXX do not use it. Special hack to use OCIO with batch API. */
GPUShader *immGetShader();
