---
icon: simple/cmake
---

# CMake options

## Frequently-used CMake options

### `CMAKE_BUILD_TYPE`

* type: STRING

Sets the build type. Possible values are `Release`, `Debug`,
`RelWithDebInfo` and `MinSizeRel`. If unset, build type is set to `Release`.

### `CMAKE_<LANG>_COMPILER`

* type: STRING

This is the command that will be used as the `<LANG>` compiler, which are `C`
and `CXX` in IREE. These variables are set to compile IREE with `clang` or
rather `clang++`. Once set, these variables can not be changed.

## IREE-specific CMake options

This gives a brief explanation of IREE specific CMake options and variables.

### `IREE_ENABLE_RUNTIME_TRACING`

* type: BOOL

Enables instrumented runtime tracing. Defaults to `OFF`.

### `IREE_ENABLE_COMPILER_TRACING`

* type: BOOL

Enables instrumented compiler tracing. This requires that
`IREE_ENABLE_RUNTIME_TRACING` also be set. Defaults to `OFF`.

### `IREE_BUILD_COMPILER`

* type: BOOL

Builds the IREE compiler. Defaults to `ON`.

### `IREE_BUILD_TESTS`

* type: BOOL

Builds IREE unit tests. Defaults to `ON`.

### `IREE_BUILD_DOCS`

* type: BOOL

Builds IREE documentation files. Defaults to `OFF`.

### `IREE_BUILD_SAMPLES`

* type: BOOL

Builds IREE sample projects. Defaults to `ON`.

### `IREE_BUILD_PYTHON_BINDINGS`

* type: BOOL

Builds the IREE python bindings. Defaults to `OFF`.

### `IREE_BUILD_BINDINGS_TFLITE`

* type: BOOL

Builds the IREE TFLite C API compatibility shim. Defaults to `ON`.

### `IREE_BUILD_BINDINGS_TFLITE_JAVA`

* type: BOOL

Builds the IREE TFLite Java bindings with the C API compatibility shim.
Defaults to `ON`.

### `IREE_BUILD_EXPERIMENTAL_REMOTING`

* type: BOOL

Builds experimental remoting component. Defaults to `OFF`.

### `IREE_HAL_DRIVER_DEFAULTS`

* type: BOOL

Default setting for each `IREE_HAL_DRIVER_*` option.

### `IREE_HAL_DRIVER_*`

* type: BOOL

Individual options enabling the build for each runtime HAL driver.

### `IREE_TARGET_BACKEND_DEFAULTS`

* type: BOOL

Default setting for each `IREE_TARGET_BACKEND_*` option.

### `IREE_TARGET_BACKEND_*`

* type: BOOL

Individual options enabling the build for each compiler target backend.

### `IREE_INPUT_*`

* type: BOOL

Individual options enabling each set of input dialects.

### `IREE_OUTPUT_FORMAT_C`

* type: BOOL

Enables the vm-c compiler output format, using MLIR EmitC. Defaults to `ON`.

### `IREE_DEV_MODE`

* type: BOOL

Configure settings to optimize for IREE development (as opposed to CI or
release). Defaults to `OFF`. For example, this will downgrade some compiler
diagnostics from errors to warnings.

### `IREE_ENABLE_LLD`

* type: BOOL

Use lld when linking. Defaults to `OFF`. This option is equivalent to
`-DIREE_USE_LINKER=lld`. The option `IREE_ENABLE_LLD` and `IREE_USE_LINKER` can
not be set at the same time.

### `IREE_ENABLE_ASAN`

* type: BOOL

Enable [address sanitizer](https://clang.llvm.org/docs/AddressSanitizer.html) if
the current build type is Debug and the compiler supports it.

### `IREE_ENABLE_MSAN`

* type: BOOL

Enable [memory sanitizer](https://clang.llvm.org/docs/MemorySanitizer.html) if
the current build type is Debug and the compiler supports it.

### `IREE_ENABLE_TSAN`

* type: BOOL

Enable [thread sanitizer](https://clang.llvm.org/docs/ThreadSanitizer.html) if
the current build type is Debug and the compiler supports it.

### `IREE_ENABLE_UBSAN`

* type: BOOL

Enable [undefiend behavior sanitizer](https://clang.llvm.org/docs/UndefinedBehaviorSanitizer.html)
if the current build type is Debug and the compiler supports it.

## Cross-compilation

When cross compiling (using a toolchain file like
[`android.toolchain.cmake`](https://android.googlesource.com/platform/ndk/+/main/build/cmake/android.toolchain.cmake)),
first build and install IREE's tools for your host configuration, then use the
`IREE_HOST_BIN_DIR` CMake option to point the cross compiled build at the
host tools.
