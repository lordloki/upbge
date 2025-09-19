/* SPDX-FileCopyrightText: 2023 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup bpygpu
 *
 * Experimental Python API, not considered public yet (called '_gpu'),
 * we may re-expose as public later.
 *
 * - Use `bpygpu_` for local API.
 * - Use `BPyGPU` for public API.
 */

#include <Python.h>

#include "GPU_context.hh"

#include "gpu_py_capabilities.hh"
#include "gpu_py_compute.hh"
#include "gpu_py_matrix.hh"
#include "gpu_py_platform.hh"
#include "gpu_py_select.hh"
#include "gpu_py_state.hh"
#include "gpu_py_types.hh"

#include "gpu_py_api.hh" /* Own include. */

#include "gpu_py_mesh_scatter.hh"

extern "C" void bpygpu_mesh_scatter_shaders_free_all(void);
static void pygpu_module_free(void *m)
{
  (void)m;
  if (GPU_context_active_get()) {
    bpygpu_mesh_scatter_shaders_free_all();
  }
}

/* -------------------------------------------------------------------- */
/** \name GPU Module
 * \{ */

PyDoc_STRVAR(
    /* Wrap. */
    pygpu_doc,
    "This module provides Python wrappers for the GPU implementation in Blender.\n"
    "Some higher level functions can be found in the :mod:`gpu_extras` module.\n");
static PyModuleDef pygpu_module_def = {
    /*m_base*/ PyModuleDef_HEAD_INIT,
    /*m_name*/ "gpu",
    /*m_doc*/ pygpu_doc,
    /*m_size*/ 0,
    /*m_methods*/ nullptr,
    /*m_slots*/ nullptr,
    /*m_traverse*/ nullptr,
    /*m_clear*/ nullptr,
    /*m_free*/ (freefunc)pygpu_module_free,
};

PyObject *BPyInit_gpu()
{
  PyObject *sys_modules = PyImport_GetModuleDict();
  PyObject *submodule;
  PyObject *mod;

  mod = PyModule_Create(&pygpu_module_def);

  PyModule_AddObject(mod, "types", (submodule = bpygpu_types_init()));
  PyDict_SetItem(sys_modules, PyModule_GetNameObject(submodule), submodule);

  PyModule_AddObject(mod, "capabilities", (submodule = bpygpu_capabilities_init()));
  PyDict_SetItem(sys_modules, PyModule_GetNameObject(submodule), submodule);

  PyModule_AddObject(mod, "matrix", (submodule = bpygpu_matrix_init()));
  PyDict_SetItem(sys_modules, PyModule_GetNameObject(submodule), submodule);

  PyModule_AddObject(mod, "platform", (submodule = bpygpu_platform_init()));
  PyDict_SetItem(sys_modules, PyModule_GetNameObject(submodule), submodule);

  PyModule_AddObject(mod, "select", (submodule = bpygpu_select_init()));
  PyDict_SetItem(sys_modules, PyModule_GetNameObject(submodule), submodule);

  PyModule_AddObject(mod, "shader", (submodule = bpygpu_shader_init()));
  PyDict_SetItem(sys_modules, PyModule_GetNameObject(submodule), submodule);

  PyModule_AddObject(mod, "state", (submodule = bpygpu_state_init()));
  PyDict_SetItem(sys_modules, PyModule_GetNameObject(submodule), submodule);

  PyModule_AddObject(mod, "texture", (submodule = bpygpu_texture_init()));
  PyDict_SetItem(sys_modules, PyModule_GetNameObject(submodule), submodule);

  PyModule_AddObject(mod, "mesh", (submodule = bpygpu_mesh_init()));
  PyDict_SetItem(sys_modules, PyModule_GetNameObject(submodule), submodule);

  PyModule_AddObject(mod, "compute", (submodule = bpygpu_compute_init()));
  PyDict_SetItem(sys_modules, PyModule_GetNameObject(submodule), submodule);

  return mod;
}

/** \} */
