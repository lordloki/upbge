/* SPDX-License-Identifier: GPL-2.0-or-later
 *
 * Minimal wrapper to run the "scatter positions -> corners + normals" compute shader
 * from Python. Inspired from BL_ArmatureObject::DoGpuSkinning (pass 2).
 */

#include <Python.h>
#include <unordered_map>
#include <mutex>

#include "BKE_idtype.hh"
#include "BKE_mesh.hh"
#include "BKE_scene.hh"

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

#include "../windowmanager/WM_api.hh"

#include "../generic/python_compat.hh" /* IWYU pragma: keep. */
#include "../intern/bpy_rna.hh"        /* pyrna_id_FromPyObject */
#include "gpu_py.hh"
#include "gpu_py_storagebuffer.hh"
#include "gpu_py_vertex_buffer.hh"

struct MeshScatterState {
  int prev_is_using_skinning = 0;
  bool requested = false;
};

struct MeshScatterResources {
  blender::gpu::Shader *shader = nullptr;
  blender::gpu::StorageBuf *ssbo_topology = nullptr;
  blender::gpu::StorageBuf *ssbo_obmat = nullptr;
  int face_offsets_offset = 0;
  int corner_to_face_offset = 0;
  int corner_verts_offset = 0;
  int vert_to_face_offsets_offset = 0;
  int vert_to_face_offset = 0;
  int normals_domain = 0;
};

/* Orphans to free later when GPU context is available. */
static std::vector<MeshScatterResources> g_mesh_scatter_orphans;
static std::unordered_map<const Mesh *, MeshScatterState> g_mesh_scatter_states;
static std::unordered_map<const Mesh *, MeshScatterResources> g_mesh_scatter_resources;
static std::mutex g_mesh_scatter_resources_mutex;

/* Free resources for a single mesh (safe to call from other translation units).
 * If no GPU context is active the resource is moved to a pending orphan list
 * and freed later (module cleanup or when a context becomes available). */
extern "C" void bpygpu_mesh_scatter_free_for_mesh(const Mesh *mesh)
{
  if (mesh == nullptr) {
    return;
  }

  /* Take ownership of the resource entry if present. */
  {
    std::lock_guard<std::mutex> lock(g_mesh_scatter_resources_mutex);
    auto it = g_mesh_scatter_resources.find(mesh);
    if (it == g_mesh_scatter_resources.end()) {
      /* Nothing to free. */
      g_mesh_scatter_states.erase(mesh);
      g_mesh_scatter_resources.erase(mesh);
      return;
    }

    /* Move resource out of the map so the map no longer contains a dangling entry. */
    MeshScatterResources res = std::move(it->second);

    /* Clean small helper maps/flags. */
    g_mesh_scatter_states.erase(mesh);
    g_mesh_scatter_resources.erase(it);

    if (GPU_context_active_get()) {
      /* Immediate GPU-safe free. */
      if (res.shader) {
        GPU_shader_free(res.shader);
      }
      if (res.ssbo_topology) {
        GPU_storagebuf_free(res.ssbo_topology);
      }
      if (res.ssbo_obmat) {
        GPU_storagebuf_free(res.ssbo_obmat);
      }
    }
    else {
      /* Defer freeing until a GPU context is available (store the resources). */
      g_mesh_scatter_orphans.push_back(std::move(res));
    }
  }
}

/* Call this from module shutdown (or periodically when a GPU context is active)
 * to flush any deferred frees. */
static void mesh_scatter_orphans_flush(void)
{
  if (!GPU_context_active_get()) {
    return;
  }
  std::lock_guard<std::mutex> lock(g_mesh_scatter_resources_mutex);
  for (MeshScatterResources &r : g_mesh_scatter_orphans) {
    if (r.shader) {
      GPU_shader_free(r.shader);
      r.shader = nullptr;
    }
    if (r.ssbo_topology) {
      GPU_storagebuf_free(r.ssbo_topology);
      r.ssbo_topology = nullptr;
    }
    if (r.ssbo_obmat) {
      GPU_storagebuf_free(r.ssbo_obmat);
      r.ssbo_obmat = nullptr;
    }
  }
  g_mesh_scatter_orphans.clear();
}

/* Create (or reuse) scatter resources for a specific mesh:
 * - builds and uploads the packed topology SSBO
 * - creates the specialized compute shader with specialization constants set to those offsets
 *
 * Returns nullptr on failure (e.g. no GPU context). */
static MeshScatterResources *mesh_scatter_resources_get_or_create(Mesh *mesh,
                                                                  int normals_domain_int)
{
  if (!mesh) {
    return nullptr;
  }

  std::lock_guard<std::mutex> lock(g_mesh_scatter_resources_mutex);

  auto it = g_mesh_scatter_resources.find(mesh);
  if (it != g_mesh_scatter_resources.end()) {
    /* Existing resources found -> return pointer to stored value. */
    return &it->second;
  }

  /* Must have a GPU context to create SSBOs / shaders. */
  if (!GPU_context_active_get()) {
    return nullptr;
  }

  MeshScatterResources res;

  /* Build topology packed ints the same way as InitStaticSkinningBuffers. */
  const auto face_offsets = mesh->face_offsets();
  const auto corner_to_face = mesh->corner_to_face_map();
  const auto corner_verts_span = mesh->corner_verts();
  std::vector<int> corner_verts_vec(corner_verts_span.begin(), corner_verts_span.end());

  const blender::OffsetIndices<int> v2f_off = mesh->vert_to_face_map_offsets();
  const blender::GroupedSpan<int> v2f = mesh->vert_to_face_map();

  const int v2f_offsets_size = v2f_off.size();
  std::vector<int> v2f_offsets(v2f_offsets_size, 0);
  for (int v = 0; v < v2f_offsets_size; ++v) {
    v2f_offsets[v] = v2f_off.data()[v];
  }
  const int total_v2f = v2f_offsets.empty() ? 0 : v2f_offsets.back();

  std::vector<int> v2f_indices;
  v2f_indices.resize(std::max(total_v2f, 0));
  if (v2f_offsets_size > 0) {
    blender::threading::parallel_for(
        blender::IndexRange(v2f_offsets_size - 1), 4096, [&](const blender::IndexRange range) {
          for (int v : range) {
            const blender::Span<int> faces_v = v2f[v];
            const int dst = v2f_off.data()[v];
            if (!faces_v.is_empty()) {
              std::copy(faces_v.begin(), faces_v.end(), v2f_indices.begin() + dst);
            }
          }
        });
  }

  /* Compute offsets for packed buffer. */
  res.face_offsets_offset = 0;
  res.corner_to_face_offset = res.face_offsets_offset + int(face_offsets.size());
  res.corner_verts_offset = res.corner_to_face_offset + int(corner_to_face.size());
  res.vert_to_face_offsets_offset = res.corner_verts_offset + int(corner_verts_vec.size());
  res.vert_to_face_offset = res.vert_to_face_offsets_offset + int(v2f_offsets.size());
  const int topo_total_size = res.vert_to_face_offset + int(v2f_indices.size());

  /* Pack into single int vector. */
  std::vector<int> topo;
  topo.reserve(topo_total_size);
  topo.insert(topo.end(), face_offsets.begin(), face_offsets.end());
  topo.insert(topo.end(), corner_to_face.begin(), corner_to_face.end());
  topo.insert(topo.end(), corner_verts_vec.begin(), corner_verts_vec.end());
  topo.insert(topo.end(), v2f_offsets.begin(), v2f_offsets.end());
  topo.insert(topo.end(), v2f_indices.begin(), v2f_indices.end());

  /* Create and upload SSBO. */
  if (!topo.empty()) {
    res.ssbo_topology = GPU_storagebuf_create(sizeof(int) * topo_total_size);
    GPU_storagebuf_update(res.ssbo_topology, topo.data());
  }

  /* Create and upload object matrix SSBO (identity matrix). */
  if (!res.ssbo_obmat) {
    float obmat[4][4];
    unit_m4(obmat);
    res.ssbo_obmat = GPU_storagebuf_create(sizeof(float) * 16);
    GPU_storagebuf_update(res.ssbo_obmat, obmat);
  }

  res.normals_domain = normals_domain_int;

  /* Create shader with specialization constants baked to mesh offsets. */
  using namespace blender::gpu::shader;
  const int group_size = 256;

  ShaderCreateInfo info("BGE_Armature_Scatter_Pass_Mesh");
  info.local_group_size(group_size, 1, 1);
  info.compute_source("draw_colormanagement_lib.glsl");
  info.storage_buf(0, Qualifier::write, "vec4", "positions[]");
  info.storage_buf(1, Qualifier::write, "uint", "normals[]");
  info.storage_buf(2, Qualifier::read, "vec4", "skinned_vert_positions[]");
  info.storage_buf(3, Qualifier::read, "mat4", "obmat[]"); /* shader expects postmat[] */
  info.storage_buf(4, Qualifier::read, "int", "topo[]");

  info.specialization_constant(Type::int_t, "face_offsets_offset", res.face_offsets_offset);
  info.specialization_constant(Type::int_t, "corner_to_face_offset", res.corner_to_face_offset);
  info.specialization_constant(Type::int_t, "corner_verts_offset", res.corner_verts_offset);
  info.specialization_constant(
      Type::int_t, "vert_to_face_offsets_offset", res.vert_to_face_offsets_offset);
  info.specialization_constant(Type::int_t, "vert_to_face_offset", res.vert_to_face_offset);
  info.specialization_constant(Type::int_t, "normals_domain", res.normals_domain);

  /* Copy the same compute_source_generated body as used in DoGpuSkinning (pass 2).
   * For brevity we reference the existing code string in this file; keep it consistent. */
  info.compute_source_generated = R"GLSL(
// Utility accessors
int face_offsets(int i) { return topo[face_offsets_offset + i]; }
int corner_to_face(int i) { return topo[corner_to_face_offset + i]; }
int corner_verts(int i) { return topo[corner_verts_offset + i]; }
int vert_to_face_offsets(int i) { return topo[vert_to_face_offsets_offset + i]; }
int vert_to_face(int i) { return topo[vert_to_face_offset + i]; }

// 10_10_10_2 packing utility
int pack_i10_trunc(float x) {
  const int signed_int_10_max = 511;
  const int signed_int_10_min = -512;
  float s = x * float(signed_int_10_max);
  int q = int(s);
  q = clamp(q, signed_int_10_min, signed_int_10_max);
  return q & 0x3FF;
}

uint pack_norm(vec3 n) {
  int nx = pack_i10_trunc(n.x);
  int ny = pack_i10_trunc(n.y);
  int nz = pack_i10_trunc(n.z);
  return uint(nx) | (uint(ny) << 10) | (uint(nz) << 20);
}

vec3 newell_face_normal_object(int f) {
  int beg = face_offsets(f);
  int end = face_offsets(f + 1);
  vec3 n = vec3(0.0);
  int v_prev_idx = corner_verts(end - 1);
  vec3 v_prev = skinned_vert_positions[v_prev_idx].xyz;
  for (int i = beg; i < end; ++i) {
    int v_curr_idx = corner_verts(i);
    vec3 v_curr = skinned_vert_positions[v_curr_idx].xyz;
    n += cross(v_prev, v_curr);
    v_prev = v_curr;
  }
  return normalize(n);
}

vec3 transform_normal(vec3 n, mat4 m) {
  return transpose(inverse(mat3(m))) * n;
}

void main() {
  uint c = gl_GlobalInvocationID.x;
  if (c >= positions.length()) {
    return;
  }

  int v = corner_verts(int(c));

  // 1) Scatter position
  vec4 p_obj = skinned_vert_positions[v];
  positions[c] = obmat[0] * p_obj;

  // 2) Calculate and scatter normal
  vec3 n_obj;
  if (normals_domain == 1) { // Face
    int f = corner_to_face(int(c));
    n_obj = newell_face_normal_object(f);
  }
  else { // Point
    int beg = vert_to_face_offsets(v);
    int end = vert_to_face_offsets(v + 1);
    vec3 n_accum = vec3(0.0);
    for (int i = beg; i < end; ++i) {
      int f = vert_to_face(i);
      n_accum += newell_face_normal_object(f);
    }
    n_obj = n_accum;
  }

  vec3 n_world = transform_normal(n_obj, obmat[0]);
  normals[c] = pack_norm(normalize(n_world));
}
)GLSL";

  blender::gpu::Shader *shader = GPU_shader_create_from_info((GPUShaderCreateInfo *)&info);
  if (!shader) {
    /* Creation failed: cleanup created ssbo if any. */
    if (res.ssbo_topology) {
      GPU_storagebuf_free(res.ssbo_topology);
      res.ssbo_topology = nullptr;
    }
    if (res.ssbo_obmat) {
      GPU_storagebuf_free(res.ssbo_obmat);
      res.ssbo_obmat = nullptr;
    }
    return nullptr;
  }

  res.shader = shader;

  /* Insert into global map and return pointer to stored value. */
  auto inserted = g_mesh_scatter_resources.emplace(mesh, std::move(res));
  return &inserted.first->second;
}

/* Free all cached mesh scatter resources (shader + ssbo) */
static void mesh_scatter_resources_free_all(void)
{
  std::lock_guard<std::mutex> lock(g_mesh_scatter_resources_mutex);
  /* free map entries as before */
  for (auto &kv : g_mesh_scatter_resources) {
    MeshScatterResources &r = kv.second;
    if (r.shader) {
      GPU_shader_free(r.shader);
      r.shader = nullptr;
    }
    if (r.ssbo_topology) {
      GPU_storagebuf_free(r.ssbo_topology);
      r.ssbo_topology = nullptr;
    }
    if (r.ssbo_obmat) {
      GPU_storagebuf_free(r.ssbo_obmat);
      r.ssbo_obmat = nullptr;
    }
  }
  g_mesh_scatter_resources.clear();

  /* Also flush orphans (if any). */
  mesh_scatter_orphans_flush();
  g_mesh_scatter_orphans.clear();
}

/* Expose C symbol for module cleanup (used from gpu module free). */
extern "C" void bpygpu_mesh_scatter_shaders_free_all(void)
{
  mesh_scatter_resources_free_all();
}

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

  Object *ob_eval = reinterpret_cast<Object *>(id_obj);
  if (!DEG_is_evaluated(ob_eval)) {
    PyErr_SetString(PyExc_TypeError, "Expected an evaluated object");
    return nullptr;
  }

  if (ob_eval->type != OB_MESH) {
    PyErr_SetString(PyExc_TypeError, "Object does not own a mesh");
    return nullptr;
  }

  Depsgraph *depsgraph = DEG_get_depsgraph_by_id(*id_obj);
  if (!depsgraph) {
    PyErr_SetString(PyExc_TypeError, "Object is not owned by a depsgraph");
    return nullptr;
  }

  /* Get evaluated object and mesh */
  if (!ob_eval) {
    PyErr_SetString(PyExc_RuntimeError, "Failed to get evaluated object");
    return nullptr;
  }

  Object *ob_orig = DEG_get_original(ob_eval);
  Mesh *mesh_orig = static_cast<Mesh *>(ob_orig->data);
  Mesh *mesh_eval = static_cast<Mesh *>(ob_eval->data);

  if (!mesh_eval || !mesh_eval->runtime || !mesh_eval->runtime->batch_cache) {
    PyErr_SetString(PyExc_RuntimeError, "Mesh batch cache not available");
    return nullptr;
  }

  /* Manage per-mesh state so modal handler can be called each frame until cache ready. */
  MeshScatterState &st = g_mesh_scatter_states[mesh_orig];

  /* Save previous flag on first encounter. */
  if (!st.requested) {
    st.prev_is_using_skinning = mesh_orig->is_using_skinning;
    mesh_orig->is_using_skinning = 1;

    /* Request geometry rebuild for that object so the draw/cache system will
     * populate VBOs (doesn't block; handled by the draw subsystem on next frame). */
    DEG_id_tag_update(&ob_orig->id, ID_RECALC_GEOMETRY);
    BKE_scene_graph_update_tagged(depsgraph, DEG_get_bmain(depsgraph));

    /* Wake UI/draw loop so next frame will run population (if applicable). */
    WM_main_add_notifier(NC_WINDOW, nullptr);

    st.requested = true;
    /* Return None for this frame; caller (modal operator) will call again next frame. */
    Py_RETURN_NONE;
  }

  /* Used to say the the object is being deformed
   * (BKE_object_is_deformed_modified) and to clear
   * tilemap shadows to avoid artifacts (Object bounds are not updated,
   * then we clear tilemaps to force shadow tilemap update) */
  mesh_eval->is_running_skinning = 1;

  /* If we already requested and cache still not ready, return None (try again next frame). */
  if (!(mesh_eval && mesh_eval->runtime && mesh_eval->runtime->batch_cache)) {
    Py_RETURN_NONE;
  }

  /* Confirm VBOs exist before proceeding; if missing, keep retrying. */
  using namespace blender::draw;
  MeshBatchCache *cache = static_cast<MeshBatchCache *>(mesh_eval->runtime->batch_cache);
  if (!cache || cache->final.buff.vbos.size() == 0) {
    Py_RETURN_NONE;
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

  /* Build / obtain the compute shader + mesh topology SSBO. */
  MeshScatterResources *res = mesh_scatter_resources_get_or_create(mesh_orig, normals_domain_int);
  if (!res || !res->shader) {
    PyErr_SetString(PyExc_RuntimeError, "Scatter compute shader not available for mesh");
    return nullptr;
  }

  GPU_storagebuf_update(res->ssbo_obmat, ob_eval->object_to_world().ptr());

  /* Prepare specialization constants state if shader expects them.
   * We only set normals_domain here; offsets must be set when topology SSBO is available.
   */
  const blender::gpu::shader::SpecializationConstants *constants_state =
      &GPU_shader_get_default_constant_state(res->shader);
  GPU_shader_bind(res->shader, constants_state);

  /* Bind destination VBOs as SSBO (these update the mesh VBOs directly) */
  vbo_pos->bind_as_ssbo(0);
  vbo_nor->bind_as_ssbo(1);

  /* Bind user SSBO (positions per vertex) at the expected binding index used by the shader */
  GPU_storagebuf_bind(py_ssbo->ssbo, 2); /* ensure shader expects skinned_vert_positions at binding 2 */

  /* Bind obmat/topo */
  GPU_storagebuf_bind(res->ssbo_obmat, 3);
  GPU_storagebuf_bind(res->ssbo_topology, 4);

  /* Dispatch groups based on number of corners */
  const int num_corners = int(mesh_eval->corner_verts().size());
  const int group_size = 256; /* match shader */
  const int num_groups_corners = (num_corners + group_size - 1) / group_size;
  GPU_compute_dispatch(res->shader, num_groups_corners, 1, 1);

  GPU_memory_barrier(GPU_BARRIER_SHADER_STORAGE | GPU_BARRIER_VERTEX_ATTRIB_ARRAY);

  GPU_shader_unbind();

  /* After dispatch : restore previous mesh flag and clear per-mesh state. */
  auto it_state = g_mesh_scatter_states.find(mesh_orig);
  if (it_state != g_mesh_scatter_states.end()) {
    mesh_orig->is_using_skinning = it_state->second.prev_is_using_skinning;
    g_mesh_scatter_states.erase(it_state);
    // Next time this mesh will be rebuild in the cache, it will be on float3.
  }

  Py_RETURN_NONE;
}
