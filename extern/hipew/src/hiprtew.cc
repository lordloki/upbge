/*
 * Copyright 2011-2023 Blender Foundation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License
 */

#include "util.h"

#include <hiprtew.h>
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>

static DynamicLibrary hiprt_lib;

#define HIPRT_LIBRARY_FIND(name) \
  name = (t##name *)dynamic_library_find(hiprt_lib, #name);

/* Function definitions. */
thiprtCreateContext *hiprtCreateContext;
thiprtDestroyContext *hiprtDestroyContext;
thiprtCreateGeometry *hiprtCreateGeometry;
thiprtDestroyGeometry *hiprtDestroyGeometry;
thiprtBuildGeometry *hiprtBuildGeometry;
thiprtGetGeometryBuildTemporaryBufferSize *hiprtGetGeometryBuildTemporaryBufferSize;
thiprtCreateScene *hiprtCreateScene;
thiprtDestroyScene *hiprtDestroyScene;
thiprtBuildScene *hiprtBuildScene;
thiprtGetSceneBuildTemporaryBufferSize *hiprtGetSceneBuildTemporaryBufferSize;
thiprtCreateFuncTable *hiprtCreateFuncTable;
thiprtSetFuncTable *hiprtSetFuncTable;
thiprtCreateGlobalStackBuffer *hiprtCreateGlobalStackBuffer;
thiprtDestroyGlobalStackBuffer *hiprtDestroyGlobalStackBuffer;
thiprtDestroyFuncTable *hiprtDestroyFuncTable;
thiprtSetLogLevel *hiprtSetLogLevel;

static DynamicLibrary dynamic_library_open_find(const char **paths) {
  int i = 0;
  while (paths[i] != NULL) {
      DynamicLibrary lib = dynamic_library_open(paths[i]);
      if (lib != NULL) {
        return lib;
      }
      ++i;
  }
  return NULL;
}

static void hipewHipRtExit(void)
{
  if (hiprt_lib != NULL) {
    /* Ignore errors. */
    dynamic_library_close(hiprt_lib);
    hiprt_lib = NULL;
  }
}

bool hiprtewInit()
{
  static bool result = false;
  static bool initialized = false;

  if (initialized) {
    return result;
  }

  initialized = true;

  if (atexit(hipewHipRtExit)) {
    return false;
  }

#ifdef _WIN32
  const char *hiprt_paths[] = {"hiprt64.dll", NULL};
#else
  /* libhiprt is installed to the bin subfolder by default, so we include it
   * in our search path. */
  const char *hiprt_paths[] = {"libhiprt64.so",
                               "/opt/rocm/lib/libhiprt64.so",
                               "/opt/rocm/bin/libhiprt64.so", NULL};
#endif

  hiprt_lib = dynamic_library_open_find(hiprt_paths);

  if (hiprt_lib == NULL) {
    return false;
  }

  HIPRT_LIBRARY_FIND(hiprtCreateContext)
  HIPRT_LIBRARY_FIND(hiprtDestroyContext)
  HIPRT_LIBRARY_FIND(hiprtCreateGeometry)
  HIPRT_LIBRARY_FIND(hiprtDestroyGeometry)
  HIPRT_LIBRARY_FIND(hiprtBuildGeometry)
  HIPRT_LIBRARY_FIND(hiprtGetGeometryBuildTemporaryBufferSize)
  HIPRT_LIBRARY_FIND(hiprtCreateScene)
  HIPRT_LIBRARY_FIND(hiprtDestroyScene)
  HIPRT_LIBRARY_FIND(hiprtBuildScene)
  HIPRT_LIBRARY_FIND(hiprtGetSceneBuildTemporaryBufferSize)
  HIPRT_LIBRARY_FIND(hiprtCreateFuncTable)
  HIPRT_LIBRARY_FIND(hiprtSetFuncTable)
  HIPRT_LIBRARY_FIND(hiprtCreateGlobalStackBuffer)
  HIPRT_LIBRARY_FIND(hiprtDestroyFuncTable)
  HIPRT_LIBRARY_FIND(hiprtDestroyGlobalStackBuffer)
  HIPRT_LIBRARY_FIND(hiprtSetLogLevel)

  result = true;

  return result;
}
