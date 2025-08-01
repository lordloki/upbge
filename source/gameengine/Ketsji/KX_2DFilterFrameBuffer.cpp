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
 * Contributor(s): Tristan Porteries.
 *
 * ***** END GPL LICENSE BLOCK *****
 */

/** \file gameengine/Ketsji/KX_2DFilterFrameBuffer.cpp
 *  \ingroup ketsji
 */

#include "KX_2DFilterFrameBuffer.h"

#include "../../blender/python/gpu/gpu_py_texture.hh"

KX_2DFilterFrameBuffer::KX_2DFilterFrameBuffer(unsigned short colorSlots,
                                               Flag flag,
                                               unsigned int width,
                                               unsigned int height)
    : RAS_2DFilterFrameBuffer(colorSlots, flag, width, height)
{
}

KX_2DFilterFrameBuffer::~KX_2DFilterFrameBuffer()
{
}

std::string KX_2DFilterFrameBuffer::GetName()
{
  return "KX_2DFilterFrameBuffer";
}

#ifdef WITH_PYTHON

PyTypeObject KX_2DFilterFrameBuffer::Type = {
    PyVarObject_HEAD_INIT(nullptr, 0) "KX_2DFilterFrameBuffer",
    sizeof(EXP_PyObjectPlus_Proxy),
    0,
    py_base_dealloc,
    0,
    0,
    0,
    0,
    py_base_repr,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    Methods,
    0,
    0,
    &EXP_Value::Type,
    0,
    0,
    0,
    0,
    0,
    0,
    py_base_new};

PyMethodDef KX_2DFilterFrameBuffer::Methods[] = {
    {"getColorTexture", (PyCFunction)KX_2DFilterFrameBuffer::sPyGetColorTexture, METH_NOARGS},
    {"getDepthTexture", (PyCFunction)KX_2DFilterFrameBuffer::sPyGetDepthTexture, METH_NOARGS},
    {nullptr, nullptr}  // Sentinel
};

PyAttributeDef KX_2DFilterFrameBuffer::Attributes[] = {
    EXP_PYATTRIBUTE_RO_FUNCTION("width", KX_2DFilterFrameBuffer, pyattr_get_width),
    EXP_PYATTRIBUTE_RO_FUNCTION("height", KX_2DFilterFrameBuffer, pyattr_get_height),
    EXP_PYATTRIBUTE_NULL  // Sentinel
};

PyObject *KX_2DFilterFrameBuffer::pyattr_get_width(EXP_PyObjectPlus *self_v,
                                                   const EXP_PYATTRIBUTE_DEF *attrdef)
{
  KX_2DFilterFrameBuffer *self = static_cast<KX_2DFilterFrameBuffer *>(self_v);
  return PyLong_FromLong(self->GetWidth());
}

PyObject *KX_2DFilterFrameBuffer::pyattr_get_height(EXP_PyObjectPlus *self_v,
                                                    const EXP_PYATTRIBUTE_DEF *attrdef)
{
  KX_2DFilterFrameBuffer *self = static_cast<KX_2DFilterFrameBuffer *>(self_v);
  return PyLong_FromLong(self->GetHeight());
}

PyObject *KX_2DFilterFrameBuffer::PyGetColorTexture(PyObject *args)
{
  blender::gpu::Texture *tex = GetColorTexture();
  if (tex) {
    return BPyGPUTexture_CreatePyObject(tex, true);
  }
  Py_RETURN_NONE;
}

PyObject *KX_2DFilterFrameBuffer::PyGetDepthTexture(PyObject *args)
{
  blender::gpu::Texture *tex = GetDepthTexture();
  if (tex) {
    return BPyGPUTexture_CreatePyObject(tex, true);
  }
  Py_RETURN_NONE;
}


#endif  // WITH_PYTHON
