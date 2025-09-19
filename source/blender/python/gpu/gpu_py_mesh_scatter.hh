/* SPDX-FileCopyrightText: 2025 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

/** \file
 * \ingroup bpygpu
 *
 * Header for the Python wrapper that runs the "scatter positions -> corners + normals"
 * compute shader from Python.
 *
 * The implementation expects a Blender Mesh with an available MeshBatchCache and
 * a user-provided SSBO containing per-vertex vec4 positions.
 */

#pragma once

#include <Python.h>

#include "BLI_compiler_attrs.h"

#include "gpu_py_storagebuffer.hh"
#include "gpu_py_vertex_buffer.hh"

/* Forward declarations of Blender types used by the implementation. */
struct Object;
struct Mesh;
struct Depsgraph;

/* -------------------------------------------------------------------- */
/** \name Python API
 *
 * Exposed to the `gpu` Python module.
 * \{ */

/**
 * Python wrapper to dispatch the compute shader that:
 *  - scatters per-vertex positions (from a user SSBO) into per-corner position VBO,
 *  - recomputes packed corner normals.
 *
 * Python signature:
 *   scatter_positions_to_corners(obj, ssbo_positions, *, normals_domain='AUTO')
 *
 * Requirements and notes:
 *  - `obj` must be convertible to a Blender `Object *` owning mesh data with a ready batch cache.
 *  - `ssbo_positions` must be a `gpu.types.GPUStorageBuf` containing `vec4` per vertex
 *    (size == verts_num * sizeof(vec4)).
 *  - A valid GPU context must be active when calling this function.
 *  - The function binds destination VBOs as SSBOs and dispatches the compute shader,
 *    then performs the required memory barriers.
 */
extern PyObject *pygpu_mesh_scatter(PyObject * self,
                                    PyObject *args,
                                    PyObject *kwds);

/**
 * Initialize the `gpu.mesh` submodule and add the scatter function.
 * Should be called during the gpu Python module initialization.
 *
 * Returns a borrowed reference to the module object on success, or nullptr on failure.
 */
extern PyObject *bpygpu_mesh_init(void) ATTR_WARN_UNUSED_RESULT;

/** \} */
