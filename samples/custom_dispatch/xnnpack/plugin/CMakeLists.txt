# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(XNNPACK_DIR "" CACHE STRING "XNNPACK directory")
set(XNNPACK_BUILD_DIR "" CACHE STRING "XNNPACK build directory")

add_library(xnnpack STATIC IMPORTED)
set_target_properties(xnnpack PROPERTIES IMPORTED_LOCATION ${XNNPACK_BUILD_DIR}/libXNNPACK.a)
set_target_properties(xnnpack PROPERTIES INTERFACE_INCLUDE_DIRECTORIES ${XNNPACK_DIR}/include)

add_library(pthreadpool STATIC IMPORTED)
set_target_properties(pthreadpool PROPERTIES IMPORTED_LOCATION ${XNNPACK_BUILD_DIR}/pthreadpool/libpthreadpool.a)
set_target_properties(pthreadpool PROPERTIES INTERFACE_INCLUDE_DIRECTORIES ${XNNPACK_BUILD_DIR}/pthreadpool-source/include)

# system-library plugin mechanism using the system dynamic library loader.
if(IREE_HAL_EXECUTABLE_PLUGIN_SYSTEM_LIBRARY)

add_library(iree_samples_custom_dispatch_xnnpack_system_plugin SHARED
  system_plugin.c
)
target_include_directories(iree_samples_custom_dispatch_xnnpack_system_plugin
  PRIVATE
    ${IREE_SOURCE_DIR}/runtime/src/
)

target_link_libraries(iree_samples_custom_dispatch_xnnpack_system_plugin xnnpack pthreadpool cpuinfo)

# NOTE: this is only required because we want this sample to run on all
# platforms without needing to change the library name (libfoo.so/foo.dll).
set_target_properties(iree_samples_custom_dispatch_xnnpack_system_plugin
  PROPERTIES
    WINDOWS_EXPORT_ALL_SYMBOLS ON
    PREFIX ""
    OUTPUT_NAME "system_plugin"
)

add_dependencies(iree-sample-deps iree_samples_custom_dispatch_xnnpack_system_plugin)

iree_lit_test_suite(
  NAME
    system_ukernel
  SRCS
    "system_ukernel.mlir"
  TOOLS
    FileCheck
    iree-compile
    iree-run-module
    iree_samples_custom_dispatch_xnnpack_system_plugin
  LABELS
    "driver=local-sync"
    "hostonly"
)

endif(IREE_HAL_EXECUTABLE_PLUGIN_SYSTEM_LIBRARY)
