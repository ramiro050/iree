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
#include <pthreadpool.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <xnnpack.h>

// The only header required from IREE:
#include "iree/hal/local/executable_plugin.h"

struct cache_key {
  size_t kernel_id;
};

struct cache_value {
  xnn_operator_t op;   // Owned by cache
  void* kernel_scale;  // Owned by cache
};

struct cache {
  size_t size;
  size_t max_size;
  struct cache_key* keys;
  struct cache_value* values;
  pthread_mutex_t lock;
};

// Stateful plugin instance.
// There may be multiple of these in a process at a time, each with its own
// load/unload pairing. We pass a pointer to this to all import calls via the
// context argument.
typedef struct {
  iree_hal_executable_plugin_allocator_t host_allocator;
  FILE* file;
  pthreadpool_t threadpool;
  struct cache* cache;
} system_plugin_t;

inline static void log_debug(const char* format, ...) {
#ifdef XNNPACK_PLUGIN_LOG
  va_list args;
  va_start(args, format);
  vfprintf(stderr, format, args);
  fprintf(stderr, "\n");
  va_end(args);
#endif
}

static struct cache* create_cache(size_t max_size) {
  struct cache* m = malloc(sizeof(struct cache));
  m->max_size = max_size;
  m->size = 0;
  m->keys = malloc(sizeof(struct cache_key) * max_size);
  m->values = malloc(sizeof(struct cache_value) * max_size);
  pthread_mutex_init(&m->lock, NULL);
  return m;
}

static void destroy_cache(struct cache* m) {
  log_debug("Max cache size used: %zu", m->size);
  for (size_t i = 0; i < m->size; i++) {
    xnn_delete_operator(m->values[i].op);
    free(m->values[i].kernel_scale);
  }
  pthread_mutex_destroy(&m->lock);
  free(m->keys);
  free(m->values);
  free(m);
}

inline static void cache_dump_key(const struct cache_key* key) {
  log_debug("cache_key { kernel_id=%zu }", key->kernel_id);
}

static bool cache_insert(struct cache* m, struct cache_key key,
                         struct cache_value value) {
  if (m->size >= m->max_size) {
    log_debug("unimplemented: variable size cache");
    return false;
  }

  log_debug("inserting entry with key:");
  cache_dump_key(&key);
  m->keys[m->size] = key;
  m->values[m->size] = value;
  m->size++;
  return true;
}

static bool cache_keys_match(const struct cache_key* key1,
                             const struct cache_key* key2) {
  return key1->kernel_id == key2->kernel_id;
}

static bool cache_lookup(const struct cache* m, const struct cache_key* key,
                         struct cache_value** value) {
  // A linear search is pretty inefficient.
  // TODO: Use more efficient search if overhead becomes large
  for (size_t i = 0; i < m->size; i++) {
    if (cache_keys_match(&m->keys[i], key)) {
      *value = &m->values[i];
      log_debug("found entry with key:");
      cache_dump_key(key);
      return true;
    }
  }
  log_debug("did not find entry with key:");
  cache_dump_key(key);
  return false;
}

static bool is_default_stride(size_t* strides, size_t* dim_sizes, size_t rank) {
  size_t acc = 1;
  for (size_t i = 0; i < rank; i++) {
    size_t index = rank - i - 1;
    if (strides[index] != acc) return false;
    acc *= dim_sizes[index];
  }
  return true;
}

static int fully_connected_nc_qd8_f32_qc4w_workgroup(void* params_ptr,
                                                     void* context,
                                                     void* reserved) {
  system_plugin_t* plugin = (system_plugin_t*)context;
  typedef struct {
    const int8_t* restrict input;
    size_t input_offset;
    size_t input_stride0;
    size_t input_stride1;
    const int8_t* restrict kernel;
    size_t kernel_offset;
    size_t kernel_stride0;
    size_t kernel_stride1;
    float* restrict output;
    size_t output_offset;
    size_t output_stride0;
    size_t output_stride1;
    size_t input_size0;
    size_t input_size1;
    size_t kernel_size0;
    size_t kernel_size1;
    size_t output_size0;
    size_t output_size1;
    int8_t transpose_rhs;
    size_t kernel_id;
  } params_t;
  const params_t* params = (const params_t*)params_ptr;

  size_t input_strides[2] = {params->input_stride0, params->input_stride1};
  size_t input_shape[2] = {params->input_size0, params->input_size1};
  if (!is_default_stride(input_strides, input_shape, /*rank=*/2)) {
    log_debug("unimplemented: input without default stride");
    log_debug("stride=[%zu, %zu], size=[%zu, %zu]", input_strides[0],
              input_strides[1], input_shape[0], input_shape[1]);
    return 1;
  }

  size_t kernel_strides[2] = {params->kernel_stride0, params->kernel_stride1};
  size_t kernel_shape[2] = {params->kernel_size0, params->kernel_size1};
  if (!is_default_stride(kernel_strides, kernel_shape, /*rank=*/2)) {
    log_debug("unimplemented: kernel without default stride");
    log_debug("stride=[%zu, %zu], size=[%zu, %zu]", kernel_strides[0],
              kernel_strides[1], kernel_shape[0], kernel_shape[1]);
    return 1;
  }

  size_t output_strides[2] = {params->output_stride0, params->output_stride1};
  size_t output_shape[2] = {params->output_size0, params->output_size1};
  if (!is_default_stride(output_strides, output_shape, /*rank=*/2)) {
    log_debug("unimplemented: output without default stride");
    log_debug("stride=[%zu, %zu], size=[%zu, %zu]", output_strides[0],
              output_strides[1], output_shape[0], output_shape[1]);
    return 1;
  }

  enum xnn_status status;
  size_t input_reduction_dim_size = input_shape[1];
  size_t kernel_reduction_dim_size =
      params->transpose_rhs ? kernel_shape[0] : kernel_shape[1];
  if (input_reduction_dim_size != kernel_reduction_dim_size) {
    log_debug("reduction dimensions are not the same size");
    return 1;
  }

  const size_t batch_size = input_shape[0];
  const size_t input_channels = input_shape[1];
  const size_t output_channels =
      params->transpose_rhs ? kernel_shape[1] : kernel_shape[0];
  if ((input_channels & 1) != 0) {
    log_debug("`input_channels` must be even");
    return 1;
  }

  // TODO: XNNPACK expects this value to be 8. From testing, this value is
  // subtracted from the kernel before using it in the matrix multiplication.
  // For IREE's use-case, this value should be 0.
  const size_t kernel_zero_point = 8;
  const float output_min = -FLT_MAX;
  const float output_max = FLT_MAX;
  // TODO: XNNPACK expects the input tensor to be padded by `XNN_EXTRA_BYTES`.
  // This padding should happen on the IREE side.
  const int8_t* input = &(params->input[params->input_offset]);
  if ((params->kernel_offset & 1) != 0) {
    log_debug("`kernel_offset` must be even");
    return 1;
  }
  const void* kernel = &(params->kernel[params->kernel_offset / 2]);
  float* output = &(params->output[params->output_offset]);

  xnn_operator_t fc_op = NULL;
  struct cache_value* cache_value_found = NULL;
  struct cache_key cache_key;
  cache_key.kernel_id = params->kernel_id;

  // The `xnn_reshape` and `xn_setup` functions modify the contents of the
  // `xnn_operator_t` passed. Therefore, when using an op from the cache the
  // cache lock must be held throughtout the entire setup, reshape, and run
  // process.
  pthread_mutex_lock(&plugin->cache->lock);
  if (cache_lookup(plugin->cache, &cache_key, &cache_value_found)) {
    fc_op = cache_value_found->op;
  } else {
    // TODO: figure out a way to avoid this allocation. From testing, passing
    // NULL seems to make the scale default to 0, which is not what we want.
    float* kernel_scale = malloc(output_channels * sizeof(float));
    for (size_t i = 0; i < output_channels; i++) kernel_scale[i] = 1;

    uint32_t flags = params->transpose_rhs ? XNN_FLAG_TRANSPOSE_WEIGHTS : 0;
    status = xnn_create_fully_connected_nc_qd8_f32_qc4w(
        input_channels, output_channels, /*input_stride=*/input_channels,
        /*output_stride=*/output_channels, kernel_zero_point, kernel_scale,
        kernel, /*bias=*/NULL, output_min, output_max, flags,
        /*code_cache=*/NULL, /*weights_cache=*/NULL, &fc_op);
    if (status != xnn_status_success) {
      log_debug("unable to create fully connected op");
      return 1;
    }

    struct cache_value cache_value;
    cache_value.op = fc_op;
    cache_value.kernel_scale = kernel_scale;
    if (!cache_insert(plugin->cache, cache_key, cache_value)) {
      log_debug("unable to cache fully connected op");
      return 1;
    }
  }

  status = xnn_reshape_fully_connected_nc_qd8_f32_qc4w(
      fc_op, batch_size,
      /*threadpool=*/plugin->threadpool);
  if (status != xnn_status_success) {
    log_debug("unable to reshape fully connected op");
    return 1;
  }

  // TODO: avoid this allocation
  struct xnn_dynamic_quantization_params* quantization_params =
      malloc(sizeof(struct xnn_dynamic_quantization_params) *
             (batch_size + XNN_EXTRA_QUANTIZATION_PARAMS));
  for (size_t i = 0; i < batch_size + XNN_EXTRA_QUANTIZATION_PARAMS; i++) {
    quantization_params[i].zero_point = 0;
    quantization_params[i].scale = 1;
  }

  status = xnn_setup_fully_connected_nc_qd8_f32_qc4w(
      fc_op, input, output,
      /*quantization_params=*/quantization_params);
  if (status != xnn_status_success) {
    log_debug("unable to setup fully connected op");
    return 1;
  }

  status = xnn_run_operator(fc_op, /*threadpool=*/plugin->threadpool);
  pthread_mutex_unlock(&plugin->cache->lock);

  if (status != xnn_status_success) {
    log_debug("unable to run fully connected op");
    return 1;
  }

  free(quantization_params);
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
  } params_t;
  const params_t* params = (const params_t*)params_ptr;
  assert(pthreadpool_get_threads_count(plugin->threadpool) == 1 &&
         "unimplemented: threadpool support");

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
  } params_t;
  const params_t* params = (const params_t*)params_ptr;
  assert(pthreadpool_get_threads_count(plugin->threadpool) == 1 &&
         "unimplemented: threadpool support");

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

  size_t thread_count = 0;
  for (size_t i = 0; i < param_count; i++) {
    if (strncmp("xnnpack-thread-count", params[i].key.data,
                params[i].key.size) == 0) {
      // `iree_hal_executable_plugin_string_t`s are not guaranteed to be null
      // terminated.
      char flag_val[params[i].value.size + 1];
      strncpy(flag_val, params[i].value.data, params[i].value.size);
      flag_val[params[i].value.size] = '\0';
      thread_count = strtol(flag_val, NULL, 10);
    }
  }
  plugin->threadpool = pthreadpool_create(thread_count);
  if (!plugin->threadpool) {
    return iree_hal_executable_plugin_status_from_code(
        IREE_HAL_EXECUTABLE_PLUGIN_STATUS_ABORTED);
  }
  // TODO: add variable size support to cache
  plugin->cache = create_cache(/*max_size=*/80);

  // Pass back the plugin instance that'll be passed to resolve.
  *out_self = plugin;
  return iree_hal_executable_plugin_ok_status();
}

// Called to free any plugin state allocated in load.
static void system_plugin_unload(void* self) {
  system_plugin_t* plugin = (system_plugin_t*)self;
  destroy_cache(plugin->cache);
  xnn_deinitialize();
  pthreadpool_destroy(plugin->threadpool);

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
