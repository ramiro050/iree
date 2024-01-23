// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Demonstrates a system linked plugin exporting a single `simple_mul_workgroup`
// function that also prints to stdout. This is not a great idea but shows how
// plugins can have side-effecting behavior - even if in most cases a standalone
// plugin can be used with much smaller code size and portability.
//
// The major use-case for a system linked plugin is JITs that may compile
// imports on-demand. Such plugins could either JIT everything on load time
// or defer JITting to the first call to a particular import. Performing JIT at
// load time is strongly preferred as it keeps all of the expensive work in one
// place before the program starts scheduling execution. Deferring will
// introduce first-run delays and require warmup steps. Since only the imports
// used by the program are present and most programs use all imports it's almost
// always going to be better to do things ahead of time.
//
// NOTE: when using the system loader all unsafe behavior is allowed: TLS,
// threads, mutable globals, syscalls, etc. Doing any of those things will
// likely break in interesting ways as the import functions are called from
// arbitrary threads concurrently. Be very careful and prefer standalone plugins
// instead except when debugging/profiling.

#include <assert.h>
#include <float.h>
#include <inttypes.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <xnnpack.h>

// The only header required from IREE:
#include "iree/hal/local/executable_plugin.h"

// Stateful plugin instance.
// There may be multiple of these in a process at a time, each with its own
// load/unload pairing. We pass a pointer to this to all import calls via the
// context argument.
typedef struct {
  iree_hal_executable_plugin_allocator_t host_allocator;
  FILE* file;
} system_plugin_t;

static int fully_connected_nc_qd8_f32_qc4w_workgroup(void* params_ptr,
                                                     void* context,
                                                     void* reserved) {
  system_plugin_t* plugin = (system_plugin_t*)context;
  typedef struct {
    const int8_t* restrict binding0;
    size_t binding0_offset;
    size_t binding0_stride0;
    size_t binding0_stride1;
    size_t binding0_stride2;
    const int8_t* restrict binding1;
    size_t binding1_offset;
    size_t binding1_stride0;
    size_t binding1_stride1;
    float* restrict binding2;
    size_t binding2_offset;
    size_t binding2_stride0;
    size_t binding2_stride1;
    size_t binding2_stride2;
    size_t binding0_size0;
    size_t binding0_size1;
    size_t binding0_size2;
    size_t binding1_size0;
    size_t binding1_size1;
    size_t binding2_size0;
    size_t binding2_size1;
    size_t binding2_size2;
    size_t threads;
  } params_t;
  const params_t* params = (const params_t*)params_ptr;

  const pthreadpool_t threadpool = pthreadpool_create(params->threads);
  assert(threadpool && "unable to create threadpool");

  enum xnn_status status;
  assert(params->binding0_size0 == 1 && "unsupported input size");
  assert(params->binding0_size2 == params->binding1_size1 &&
         "invalid fully connected reduction");

  const size_t batch_size = params->binding0_size1;
  const size_t input_channels = params->binding0_size2;
  const size_t output_channels = params->binding1_size0;
  assert((input_channels & 1) == 0 && "`input_channels` must be even");
  // TODO: XNNPACK expects this value to be 8. From testing, this value is
  // subtracted from the kernel before using it in the matrix multiplication.
  // For IREE's use-case, this value should be 0.
  const size_t kernel_zero_point = 8;
  const float output_min = -FLT_MAX;
  const float output_max = FLT_MAX;
  // TODO: XNNPACK expects the input tensor to be padded by `XNN_EXTRA_BYTES`.
  // This padding should happen on the IREE side.
  const int8_t* input = &(params->binding0[params->binding0_offset]);
  // TODO: handle sub-byte offset
  assert(params->binding1_offset == 0 &&
         "unimplemented: non-zero offset for kernel buffer");
  const void* kernel = params->binding1;
  float* output = &(params->binding2[params->binding2_offset]);

  // TODO: figure out a way to avoid this allocation. From testing, passing NULL
  // seems to make the scale default to 0, which is not what we want.
  float* kernel_scale = malloc(output_channels * sizeof(float));
  for (size_t i = 0; i < output_channels; i++) kernel_scale[i] = 1;

  xnn_operator_t fc_op = NULL;
  status = xnn_create_fully_connected_nc_qd8_f32_qc4w(
      input_channels, output_channels, /*input_stride=*/input_channels,
      /*output_stride=*/output_channels, kernel_zero_point, kernel_scale,
      kernel, /*bias=*/NULL, output_min, output_max, /*flags=*/0,
      /*code_cache=*/NULL,
      /*weights_cache=*/NULL, &fc_op);
  assert(status == xnn_status_success && "unable to create fully connected op");
  status =
      xnn_reshape_fully_connected_nc_qd8_f32_qc4w(fc_op, batch_size,
                                                  /*threadpool=*/threadpool);
  assert(status == xnn_status_success &&
         "unable to reshape fully connected op");

  // TODO: avoid this allocation
  struct xnn_dynamic_quantization_params* quantization_params =
      malloc(sizeof(struct xnn_dynamic_quantization_params) *
             (batch_size + XNN_EXTRA_QUANTIZATION_PARAMS));
  for (size_t i = 0; i < batch_size + XNN_EXTRA_QUANTIZATION_PARAMS; i++) {
    quantization_params[i].zero_point = 0;
    quantization_params[i].scale = 1;
  }
  status = xnn_setup_fully_connected_nc_qd8_f32_qc4w(
      fc_op, input, output, /*quantization_params=*/quantization_params);
  assert(status == xnn_status_success && "unable to setup fully connected op");

  status = xnn_run_operator(fc_op, /*threadpool=*/threadpool);
  assert(status == xnn_status_success && "unable to run fully connected op");

  free(quantization_params);
  free(kernel_scale);
  pthreadpool_destroy(threadpool);
  return 0;
}

static int multiply2_1d_workgroup(void* params_ptr, void* context,
                                  void* reserved) {
  system_plugin_t* plugin = (system_plugin_t*)context;
  typedef struct {
    const float* restrict binding0;
    size_t binding0_offset;
    size_t binding0_stride;
    const float* restrict binding1;
    size_t binding1_offset;
    size_t binding1_stride;
    float* restrict binding2;
    size_t binding2_offset;
    size_t binding2_stride;
    size_t binding0_size;
    size_t binding1_size;
    size_t binding2_size;
    size_t threads;
  } params_t;
  const params_t* params = (const params_t*)params_ptr;
  assert(params->threads == 1 && "unimplemented: threadpool support");

  enum xnn_status status;
  xnn_subgraph_t subgraph = NULL;
  status =
      xnn_create_subgraph(/*external_value_ids=*/3, /*flags=*/0, &subgraph);
  assert(status == xnn_status_success && "unable to create subgraph");

  assert(params->binding0_size == params->binding1_size &&
         params->binding0_size == params->binding2_size &&
         "unimplemented: broadcasting");
  const size_t dims[1] = {params->binding0_size};
  uint32_t lhs_id = XNN_INVALID_VALUE_ID;
  status = xnn_define_tensor_value(subgraph, /*datatype=*/xnn_datatype_fp32,
                                   /*num_dims=*/1, /*dims=*/dims,
                                   /*data=*/NULL,
                                   /*external_id=*/0,
                                   /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT,
                                   /*id_out=*/&lhs_id);
  assert(status == xnn_status_success && "unable to define lhs input tensor");

  uint32_t rhs_id = XNN_INVALID_VALUE_ID;
  status = xnn_define_tensor_value(subgraph, /*datatype=*/xnn_datatype_fp32,
                                   /*num_dims=*/1, /*dims=*/dims,
                                   /*data=*/NULL,
                                   /*external_id=*/1,
                                   /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT,
                                   /*id_out=*/&rhs_id);
  assert(status == xnn_status_success && "unable to define rhs input tensor");

  uint32_t output_id = XNN_INVALID_VALUE_ID;
  status = xnn_define_tensor_value(subgraph, xnn_datatype_fp32,
                                   /*num_dims=*/1, dims, NULL,
                                   /*external_id=*/2,
                                   XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id);
  assert(status == xnn_status_success && "unable to define output tensor");

  status = xnn_define_multiply2(subgraph, -100.0f, 100.0f, lhs_id, rhs_id,
                                output_id, /*flags=*/0);
  assert(status == xnn_status_success && "unable to define multiply2");

  xnn_runtime_t runtime = NULL;
  status = xnn_create_runtime(subgraph, &runtime);
  assert(status == xnn_status_success && "unable to create runtime");
  struct xnn_external_value lhs_external_value = {
      lhs_id, (void*)&(params->binding0[params->binding0_offset])};
  struct xnn_external_value rhs_external_value = {
      rhs_id, (void*)&(params->binding1[params->binding1_offset])};
  struct xnn_external_value output_external_value = {
      output_id, (void*)&(params->binding2[params->binding2_offset])};
  const struct xnn_external_value externals[3] = {
      lhs_external_value, rhs_external_value, output_external_value};
  status = xnn_setup_runtime(runtime, /*num_external_values=*/3, externals);
  assert(status == xnn_status_success && "unable to setup runtime");

  status = xnn_invoke_runtime(runtime);
  assert(status == xnn_status_success && "unable to invoke runtime");

  status = xnn_delete_runtime(runtime);
  assert(status == xnn_status_success && "unable to delete runtime");
  status = xnn_delete_subgraph(subgraph);
  assert(status == xnn_status_success && "unable to delete subgraph");

  for (size_t i = 0; i < params->binding0_size; ++i) {
    float curr_lhs = params->binding0[params->binding0_offset + i];
    float curr_rhs = params->binding1[params->binding1_offset + i];
    float curr_output = params->binding2[params->binding2_offset + i];
    fprintf(plugin->file, "mul2[%zu](%g * %g = %g)\n", i, curr_lhs, curr_rhs,
            curr_output);
  }

  return 0;
}

// TODO(ramiro050): refactor common logic with multiply2 above
static int batch_matrix_multiply_workgroup(void* params_ptr, void* context,
                                           void* reserved) {
  system_plugin_t* plugin = (system_plugin_t*)context;
  typedef struct {
    const float* restrict binding0;
    size_t binding0_offset;
    size_t binding0_stride0;
    size_t binding0_stride1;
    size_t binding0_stride2;
    const float* restrict binding1;
    size_t binding1_offset;
    size_t binding1_stride0;
    size_t binding1_stride1;
    size_t binding1_stride2;
    float* restrict binding2;
    size_t binding2_offset;
    size_t binding2_stride0;
    size_t binding2_stride1;
    size_t binding2_stride2;
    size_t binding0_size0;
    size_t binding0_size1;
    size_t binding0_size2;
    size_t binding1_size0;
    size_t binding1_size1;
    size_t binding1_size2;
    size_t binding2_size0;
    size_t binding2_size1;
    size_t binding2_size2;
    size_t threads;
  } params_t;
  const params_t* params = (const params_t*)params_ptr;
  assert(params->threads == 1 && "unimplemented: threadpool support");

  enum xnn_status status;
  xnn_subgraph_t subgraph = NULL;
  status =
      xnn_create_subgraph(/*external_value_ids=*/3, /*flags=*/0, &subgraph);
  assert(status == xnn_status_success && "unable to create subgraph");

  assert(params->binding0_size1 == params->binding1_size0 && "invalid matmul");
  const size_t rank = 3;
  const size_t dims0[] = {params->binding0_size0, params->binding0_size1,
                          params->binding0_size2};
  const size_t dims1[] = {params->binding1_size0, params->binding1_size1,
                          params->binding1_size2};
  const size_t dims2[] = {params->binding2_size0, params->binding2_size1,
                          params->binding2_size2};

  uint32_t lhs_id = XNN_INVALID_VALUE_ID;
  status = xnn_define_tensor_value(subgraph, /*datatype=*/xnn_datatype_fp32,
                                   /*num_dims=*/rank, /*dims=*/dims0,
                                   /*data=*/NULL,
                                   /*external_id=*/0,
                                   /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT,
                                   /*id_out=*/&lhs_id);
  assert(status == xnn_status_success && "unable to define lhs input tensor");

  uint32_t rhs_id = XNN_INVALID_VALUE_ID;
  status = xnn_define_tensor_value(subgraph, /*datatype=*/xnn_datatype_fp32,
                                   /*num_dims=*/rank, /*dims=*/dims1,
                                   /*data=*/NULL,
                                   /*external_id=*/1,
                                   /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT,
                                   /*id_out=*/&rhs_id);
  assert(status == xnn_status_success && "unable to define rhs input tensor");

  uint32_t output_id = XNN_INVALID_VALUE_ID;
  status = xnn_define_tensor_value(subgraph, xnn_datatype_fp32,
                                   /*num_dims=*/rank, dims2, NULL,
                                   /*external_id=*/2,
                                   XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id);
  assert(status == xnn_status_success && "unable to define output tensor");

  status = xnn_define_batch_matrix_multiply(subgraph, lhs_id, rhs_id, output_id,
                                            /*flags=*/0);
  assert(status == xnn_status_success && "unable to define multiply2");

  xnn_runtime_t runtime = NULL;
  status = xnn_create_runtime(subgraph, &runtime);
  assert(status == xnn_status_success && "unable to create runtime");
  struct xnn_external_value lhs_external_value = {
      lhs_id, (void*)&(params->binding0[params->binding0_offset])};
  struct xnn_external_value rhs_external_value = {
      rhs_id, (void*)&(params->binding1[params->binding1_offset])};
  struct xnn_external_value output_external_value = {
      output_id, (void*)&(params->binding2[params->binding2_offset])};
  const struct xnn_external_value externals[3] = {
      lhs_external_value, rhs_external_value, output_external_value};
  status = xnn_setup_runtime(runtime, /*num_external_values=*/3, externals);
  assert(status == xnn_status_success && "unable to setup runtime");

  status = xnn_invoke_runtime(runtime);
  assert(status == xnn_status_success && "unable to invoke runtime");

  status = xnn_delete_runtime(runtime);
  assert(status == xnn_status_success && "unable to delete runtime");
  status = xnn_delete_subgraph(subgraph);
  assert(status == xnn_status_success && "unable to delete subgraph");
  return 0;
}

// Called once for each plugin load and paired with a future call to unload.
// Even in standalone mode we could allocate using environment->host_allocator,
// set an out_self pointer, and parse parameters but here in system mode we can
// do whatever we want.
//
// If any state is required it should be allocated and stored in |out_self|.
// This self value will be passed to all future calls related to the particular
// instance. Note that there may be multiple instances of a plugin in any
// particular process and this must be thread-safe.
static iree_hal_executable_plugin_status_t system_plugin_load(
    const iree_hal_executable_plugin_environment_v0_t* environment,
    size_t param_count, const iree_hal_executable_plugin_string_pair_t* params,
    void** out_self) {
  const struct xnn_allocator* allocator = NULL;
  enum xnn_status xnn_status = xnn_initialize(allocator);
  if (xnn_status != xnn_status_success) {
    return iree_hal_executable_plugin_status_from_code(
        IREE_HAL_EXECUTABLE_PLUGIN_STATUS_ABORTED);
  }
  // Allocate the plugin state.
  system_plugin_t* plugin = NULL;
  iree_hal_executable_plugin_status_t status =
      iree_hal_executable_plugin_allocator_malloc(
          environment->host_allocator, sizeof(*plugin), (void**)&plugin);
  if (status) return status;
  plugin->host_allocator = environment->host_allocator;

  // "Open standard out" simulating us doing some syscalls or other expensive
  // stateful/side-effecting things.
  plugin->file = stdout;

  // Pass back the plugin instance that'll be passed to resolve.
  *out_self = plugin;
  return iree_hal_executable_plugin_ok_status();
}

// Called to free any plugin state allocated in load.
static void system_plugin_unload(void* self) {
  xnn_deinitialize();
  system_plugin_t* plugin = (system_plugin_t*)self;
  iree_hal_executable_plugin_allocator_t host_allocator =
      plugin->host_allocator;

  // "Close standard out" simulating us doing some syscalls and other expensive
  // stateful/side-effecting things.
  fflush(plugin->file);
  plugin->file = NULL;

  // Free the plugin state using the same allocator it came from.
  iree_hal_executable_plugin_allocator_free(host_allocator, plugin);
}

// Called to resolve one or more imports by symbol name.
// See the plugin API header for more information. Note that some of the
// functions may already be resolved and some may be optional.
static iree_hal_executable_plugin_status_t system_plugin_resolve(
    void* self, const iree_hal_executable_plugin_resolve_params_v0_t* params,
    iree_hal_executable_plugin_resolution_t* out_resolution) {
  system_plugin_t* plugin = (system_plugin_t*)self;
  *out_resolution = 0;
  bool any_required_not_found = false;
  for (size_t i = 0; i < params->count; ++i) {
    if (params->out_fn_ptrs[i]) continue;
    const char* symbol_name = params->symbol_names[i];
    bool is_optional =
        iree_hal_executable_plugin_import_is_optional(symbol_name);
    if (is_optional) ++symbol_name;
    if (iree_hal_executable_plugin_strcmp(symbol_name,
                                          "xnnpack.multiply2_workgroup") == 0) {
      params->out_fn_ptrs[i] = multiply2_1d_workgroup;
      params->out_fn_contexts[i] =
          plugin;  // passing plugin to each import call
    } else if (iree_hal_executable_plugin_strcmp(
                   symbol_name, "xnnpack.batch_matrix_multiply_workgroup") ==
               0) {
      params->out_fn_ptrs[i] = batch_matrix_multiply_workgroup;
      params->out_fn_contexts[i] =
          plugin;  // passing plugin to each import call
    } else if (iree_hal_executable_plugin_strcmp(
                   symbol_name,
                   "xnnpack.fully_connected_nc_qd8_f32_qc4w_workgroup") == 0) {
      params->out_fn_ptrs[i] = fully_connected_nc_qd8_f32_qc4w_workgroup;
      params->out_fn_contexts[i] =
          plugin;  // passing plugin to each import call
    } else {
      if (is_optional) {
        *out_resolution |=
            IREE_HAL_EXECUTABLE_PLUGIN_RESOLUTION_MISSING_OPTIONAL;
      } else {
        any_required_not_found = true;
      }
    }
  }
  return any_required_not_found
             ? iree_hal_executable_plugin_status_from_code(
                   IREE_HAL_EXECUTABLE_PLUGIN_STATUS_NOT_FOUND)
             : iree_hal_executable_plugin_ok_status();
}

// Exported on the shared library and used by the runtime to query the plugin
// interface. When statically linking the plugin this is just a function that
// can be called and can have any name to allow for multiple plugins. When
// dynamically linking the exported symbol must be exactly this with no C++
// name mangling.
IREE_HAL_EXECUTABLE_PLUGIN_EXPORT const iree_hal_executable_plugin_header_t**
iree_hal_executable_plugin_query(
    iree_hal_executable_plugin_version_t max_version, void* reserved) {
  static const iree_hal_executable_plugin_header_t header = {
      // Declares what library version is present: newer runtimes may support
      // loading older plugins but newer plugins cannot load on older runtimes.
      .version = IREE_HAL_EXECUTABLE_PLUGIN_VERSION_LATEST,
      // Name and description are used for tracing/logging/diagnostics.
      .name = "sample_system",
      .description =
          "system plugin sample "
          "(custom_dispatch/cpu/plugin/system_plugin.c)",
      .features = 0,
      // Let the runtime know what sanitizer this plugin was compiled with.
      .sanitizer = IREE_HAL_EXECUTABLE_PLUGIN_SANITIZER_KIND,
  };
  static const iree_hal_executable_plugin_v0_t plugin = {
      .header = &header,
      .load = system_plugin_load,
      .unload = system_plugin_unload,
      .resolve = system_plugin_resolve,
  };
  return max_version <= IREE_HAL_EXECUTABLE_PLUGIN_VERSION_LATEST
             ? (const iree_hal_executable_plugin_header_t**)&plugin
             : NULL;
}
