# ***** BEGIN GPL LICENSE BLOCK *****
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
#
# The Original Code is Copyright (C) 2006, Blender Foundation
# All rights reserved.
#
# The Original Code is: all of this file.
#
# Contributor(s): Jacques Beaurain.
#
# ***** END GPL LICENSE BLOCK *****

set(INC
  .
  KXNetwork
  ../BlenderRoutines
  ../Common
  ../Converter
  ../Device
  ../Expressions
  ../GameLogic
  ../Launcher
  ../Physics/Common
  ../Rasterizer
  ../Rasterizer/RAS_OpenGLRasterizer
  ../SceneGraph
  ../VideoTexture
  ../../blender
  ../../blender/blenfont
  ../../blender/blenkernel
  ../../blender/blenlib
  ../../blender/blenloader
  ../../blender/blentranslation
  ../../blender/draw
  ../../blender/draw/intern
  ../../blender/draw/engines/eevee
  ../../blender/editors/include
  ../../blender/gpu
  ../../blender/imbuf
  ../../blender/makesdna
  ../../blender/makesrna
  ../../blender/python
  ../../blender/python/generic
  ../../blender/python/mathutils
  ../../blender/python/intern
  ../../blender/render/
  ../../blender/render/intern/
  ../../blender/windowmanager
  ../../../intern/ghost
  ../../../intern/glew-mx
  ../../../intern/guardedalloc
  ../../../intern/string
)

set(INC_SYS
  ../../../intern/moto/include
  ../../../extern/recastnavigation/Recast/Include
  ../../../extern/recastnavigation/Detour/Include
  ${PTHREADS_INCLUDE_DIRS}
  ${GLEW_INCLUDE_PATH}
  ${BOOST_INCLUDE_DIR}
)

set(SRC
  BL_Action.cpp
  BL_ActionManager.cpp
  BL_Shader.cpp
  BL_Texture.cpp
  KX_2DFilter.cpp
  KX_2DFilterManager.cpp
  KX_2DFilterFrameBuffer.cpp
  KX_BlenderCanvas.cpp
  KX_BlenderMaterial.cpp
  KX_Camera.cpp
  KX_CameraIpoSGController.cpp
  KX_CharacterWrapper.cpp
  KX_CollisionEventManager.cpp
  KX_ConstraintWrapper.cpp
  KX_EmptyObject.cpp
  KX_FontObject.cpp
  KX_GameObject.cpp
  KX_Globals.cpp
  KX_IPO_SGController.cpp
  KX_KetsjiEngine.cpp
  KX_LibLoadStatus.cpp
  KX_Light.cpp
  KX_LightIpoSGController.cpp
  KX_LodLevel.cpp
  KX_LodManager.cpp
  KX_MaterialIpoController.cpp
  KX_MaterialShader.cpp
  KX_MeshProxy.cpp
  KX_MotionState.cpp
  KX_NavMeshObject.cpp
  KX_ObColorIpoSGController.cpp
  KX_ObstacleSimulation.cpp
  KX_OrientationInterpolator.cpp
  KX_PolyProxy.cpp
  KX_PositionInterpolator.cpp
  KX_PyConstraintBinding.cpp
  KX_PyMath.cpp
  KX_PythonComponent.cpp
  KX_PythonComponentManager.cpp
  KX_PythonInit.cpp
  KX_PythonInitTypes.cpp
  KX_PythonMain.cpp
  KX_RayCast.cpp
  KX_SG_BoneParentNodeRelationship.cpp
  KX_SG_NodeRelationships.cpp
  KX_ScalarInterpolator.cpp
  KX_ScalingInterpolator.cpp
  KX_Scene.cpp
  KX_TimeCategoryLogger.cpp
  KX_TimeLogger.cpp
  KX_VehicleWrapper.cpp
  KX_VertexProxy.cpp
  KX_CollisionContactPoints.cpp

  BL_Action.h
  BL_ActionManager.h
  BL_Shader.h
  BL_Texture.h
  KX_2DFilter.h
  KX_2DFilterManager.h
  KX_2DFilterFrameBuffer.h
  KX_BlenderCanvas.h
  KX_BlenderMaterial.h
  KX_Camera.h
  KX_CameraIpoSGController.h
  KX_CharacterWrapper.h
  KX_ClientObjectInfo.h
  KX_ConstraintWrapper.h
  KX_EmptyObject.h
  KX_FontObject.h
  KX_GameObject.h
  KX_Globals.h
  KX_IInterpolator.h
  KX_IPOTransform.h
  KX_IPO_SGController.h
  KX_IScalarInterpolator.h
  KX_ISystem.h
  KX_KetsjiEngine.h
  KX_LibLoadStatus.h
  KX_Light.h
  KX_LightIpoSGController.h
  KX_LodLevel.h
  KX_LodManager.h
  KX_MaterialIpoController.h
  KX_MaterialShader.h
  KX_MeshProxy.h
  KX_MotionState.h
  KX_NavMeshObject.h
  KX_ObColorIpoSGController.h
  KX_ObstacleSimulation.h
  KX_OrientationInterpolator.h
  KX_PhysicsEngineEnums.h
  KX_PolyProxy.h
  KX_PositionInterpolator.h
  KX_PyConstraintBinding.h
  KX_PyMath.h
  KX_PythonComponent.h
  KX_PythonComponentManager.h
  KX_PythonInit.h
  KX_PythonInitTypes.h
  KX_PythonMain.h
  KX_RayCast.h
  KX_SG_BoneParentNodeRelationship.h
  KX_SG_NodeRelationships.h
  KX_ScalarInterpolator.h
  KX_ScalingInterpolator.h
  KX_Scene.h
  KX_TimeCategoryLogger.h
  KX_TimeLogger.h
  KX_CollisionEventManager.h
  KX_VehicleWrapper.h
  KX_VertexProxy.h
  KX_CollisionContactPoints.h
)

set(LIB
  ge_converter
  ge_logic_bricks
  ge_scenegraph
  ge_msg_network
  ge_device
  extern_recastnavigation
  bf_blenkernel
  ge_rasterizer
)

add_definitions(${GL_DEFINITIONS})

if(WITH_IMAGE_DDS)
  add_definitions(-DWITH_DDS)
endif()

if(WITH_SDL)
  list(APPEND INC_SYS
    ${SDL_INCLUDE_DIR}
  )
  add_definitions(-DWITH_SDL)
endif()

if(WITH_CODEC_FFMPEG)
  add_definitions(-DWITH_FFMPEG)
endif()

if(WITH_BULLET)
  list(APPEND INC
    ../Physics/Bullet
  )
  list(APPEND INC_SYS
    ${BULLET_INCLUDE_DIRS}
  )
  add_definitions(-DWITH_BULLET)
endif()

if(WITH_AUDASPACE)
  list(APPEND INC_SYS
    ${AUDASPACE_C_INCLUDE_DIRS}
  )
  list(APPEND LIB
    ${AUDASPACE_C_LIBRARIES}
    ${AUDASPACE_PY_LIBRARIES}
  )
  add_definitions(-DWITH_AUDASPACE)
endif()

if(WITH_GAMEENGINE)
  add_definitions(-DWITH_GAMEENGINE)
endif()

if(WITH_CYCLES)
  add_definitions(-DWITH_CYCLES)
endif()

if(WITH_MOD_FLUID)
  list(APPEND LIB
    bf_intern_mantaflow
  )
  add_definitions(-DWITH_FLUID)
endif()

blender_add_lib(ge_ketsji "${SRC}" "${INC}" "${INC_SYS}" "${LIB}")
