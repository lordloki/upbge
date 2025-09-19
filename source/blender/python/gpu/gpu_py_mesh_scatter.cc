/* SPDX-License-Identifier: GPL-2.0-or-later
 *
 * Minimal wrapper to run the "scatter positions -> corners + normals" compute shader
 * from Python. Inspired from BL_ArmatureObject::DoGpuSkinning (pass 2).
 */

#include <Python.h>

#include "BKE_idtype.hh"
#include "BKE_mesh.hh"

#include "BLI_string_utf8.h"

#include "DNA_ID.h"
#include "DNA_mesh_types.h"

#include "../draw/intern/draw_cache_extract.hh"
#include "../gpu/intern/gpu_shader_create_info.hh"

#include "GPU_context.hh"
#include "GPU_compute.hh"
#include "GPU_shader.hh"
#include "GPU_state.hh"
#include "GPU_storage_buffer.hh"
#include "GPU_vertex_buffer.hh"

#include "../depsgraph/DEG_depsgraph_query.hh"

#include "../generic/python_compat.hh" /* IWYU pragma: keep. */
#include "../intern/bpy_rna.hh"        /* pyrna_id_FromPyObject */
#include "gpu_py.hh"
#include "gpu_py_storagebuffer.hh"
#include "gpu_py_vertex_buffer.hh"

/* Helper: get MeshBatchCache and vbos (adapt to your project paths) */
/* TODO: include the correct header(s) to access MeshBatchCache / VBOType / lookup_ptr. */
/* extern MeshBatchCache *DRW_mesh_batch_cache_get(Mesh *me); */ /* adapt if needed */

PyDoc_STRVAR(pygpu_mesh_scatter_doc,
             ".. function:: scatter_positions_to_corners(obj, ssbo_positions)\n"
             "\n"
             "   Scatter per-vertex positions (from user SSBO) to per-corner VBOs and recompute\n"
             "   packed normals using the internal compute shader. The mesh VBOs (positions and\n"
             "   normals) will be updated and ready for rendering.\n\n"
             "   `obj` must be an evaluated bpy.types.Object owning a mesh. `ssbo_positions`\n"
             "   must be a gpu.types.GPUStorageBuf containing vec4 per vertex.\n");

static PyObject *pygpu_mesh_scatter(PyObject * /*self*/, PyObject *args, PyObject *kwds)
{
  PyObject *py_obj = nullptr;
  BPyGPUStorageBuf *py_ssbo = nullptr;

  static const char *_keywords[] = {"obj", "ssbo", nullptr};
  if (!PyArg_ParseTupleAndKeywords(
          args, kwds, "OO:scatter_positions_to_corners", (char **)_keywords, &py_obj, &py_ssbo))
  {
    return nullptr;
  }

  /* Validate GPU context */
  if (!GPU_context_active_get()) {
    PyErr_SetString(PyExc_RuntimeError, "No active GPU context");
    return nullptr;
  }

  /* Validate ssbo */
  if (!py_ssbo || py_ssbo->ssbo == nullptr) {
    PyErr_SetString(PyExc_TypeError, "Expected a GPUStorageBuf as second argument");
    return nullptr;
  }

  /* Convert Python object to Blender Object* using pyrna helper (pattern from
   * bpy_geometry_set.cc). */
  ID *id_obj = nullptr;
  if (!pyrna_id_FromPyObject(py_obj, &id_obj)) {
    PyErr_Format(PyExc_TypeError, "Expected an Object, not %.200s", Py_TYPE(py_obj)->tp_name);
    return nullptr;
  }

  if (GS(id_obj->name) != ID_OB) {
    PyErr_Format(PyExc_TypeError,
                 "Expected an Object, not %.200s",
                 BKE_idtype_idcode_to_name(GS(id_obj->name)));
    return nullptr;
  }

  Object *evaluated_object = reinterpret_cast<Object *>(id_obj);
  if (!DEG_is_evaluated(evaluated_object)) {
    PyErr_SetString(PyExc_TypeError, "Expected an evaluated object");
    return nullptr;
  }

  Depsgraph *depsgraph = DEG_get_depsgraph_by_id(*id_obj);
  if (!depsgraph) {
    PyErr_SetString(PyExc_TypeError, "Object is not owned by a depsgraph");
    return nullptr;
  }

  /* Get evaluated object and mesh */
  Object *ob_eval = DEG_get_evaluated(depsgraph, evaluated_object);
  if (!ob_eval) {
    PyErr_SetString(PyExc_RuntimeError, "Failed to get evaluated object");
    return nullptr;
  }

  Mesh *mesh_eval = static_cast<Mesh *>(ob_eval->data);
  if (!mesh_eval || !mesh_eval->runtime || !mesh_eval->runtime->batch_cache) {
    PyErr_SetString(PyExc_RuntimeError, "Mesh batch cache not available");
    return nullptr;
  }

  using namespace blender::draw;
  MeshBatchCache *cache = static_cast<MeshBatchCache *>(mesh_eval->runtime->batch_cache);
  if (!cache || cache->final.buff.vbos.size() == 0) {
    PyErr_SetString(PyExc_RuntimeError, "Mesh VBOs not ready");
    return nullptr;
  }

  /* Lookup vbos */
  blender::gpu::VertBuf *vbo_pos = nullptr;
  blender::gpu::VertBuf *vbo_nor = nullptr;
  auto pos_it = cache->final.buff.vbos.lookup_ptr(VBOType::Position);
  if (pos_it) {
    vbo_pos = pos_it->get();
  }
  auto nor_it = cache->final.buff.vbos.lookup_ptr(VBOType::CornerNormal);
  if (nor_it) {
    vbo_nor = nor_it->get();
  }
  if (!vbo_pos || !vbo_nor) {
    PyErr_SetString(PyExc_RuntimeError, "Required VBOs not present in cache");
    return nullptr;
  }

  /* Determine normals_domain automatically from evaluated mesh. */
  const int normals_domain_int = (mesh_eval->normals_domain() ==
                                  blender::bke::MeshNormalDomain::Face) ?
                                     1 :
                                     0;

  /* Prepare topology SSBOs + offsets exactly like InitStaticSkinningBuffers.
   * For reuse you can cache these per-mesh. Here we assume they exist or will be created.
   * TODO: implement creation / retrieval of:
   *   - ssbo_topology (packed ints)
   *   - ssbo_postmat (mat4)
   *   - specialization constants: face_offsets_offset, corner_to_face_offset, ...
   */
  blender::gpu::StorageBuf *ssbo_topo = nullptr;    /* TODO: create or fetch */
  blender::gpu::StorageBuf *ssbo_obmat = nullptr; /* TODO: create or fetch */

  /* Build / obtain the compute shader.
   * TODO: reuse a shared shader instance or create one here using the GLSL from BL_ArmatureObject.
   * The shader must declare specialization constants for offsets and for normals_domain.
   */
  blender::gpu::Shader *scatter_shader = nullptr; /* TODO: create or fetch shader */

  if (!scatter_shader) {
    PyErr_SetString(PyExc_RuntimeError,
                    "Scatter compute shader not available (TODO: create shader from source)");
    return nullptr;
  }

  /* Prepare specialization constants state if shader expects them.
   * We only set normals_domain here; offsets must be set when topology SSBO is available.
   */
  blender::gpu::shader::SpecializationConstants spec_consts;

  /* Bind destination VBOs as SSBO (these update the mesh VBOs directly) */
  vbo_pos->bind_as_ssbo(0);
  vbo_nor->bind_as_ssbo(1);

  /* Bind user SSBO (positions per vertex) at the expected binding index used by the shader */
  GPU_storagebuf_bind(py_ssbo->ssbo, 2); /* ensure shader expects skinned_vert_positions at binding 2 */

  /* Bind postmat/topo (must be provided/created by TODO above) */
  GPU_storagebuf_bind(ssbo_obmat, 3);
  GPU_storagebuf_bind(ssbo_topo, 4);

  /* Dispatch groups based on number of corners */
  const int num_corners = int(mesh_eval->corner_verts().size());
  const int group_size = 256; /* match shader */
  const int num_groups_corners = (num_corners + group_size - 1) / group_size;
  GPU_compute_dispatch(scatter_shader, num_groups_corners, 1, 1);

  GPU_memory_barrier(GPU_BARRIER_SHADER_STORAGE | GPU_BARRIER_VERTEX_ATTRIB_ARRAY);

  GPU_shader_unbind();

  Py_RETURN_NONE;
}
