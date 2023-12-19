// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/io/parameter_index.h"

#include "iree/base/internal/atomics.h"
#include "iree/base/internal/synchronization.h"

struct iree_io_parameter_index_t {
  iree_atomic_ref_count_t ref_count;
  iree_allocator_t host_allocator;

  // Guards mutation of the entries list.
  // NOTE: this does not guard the entries themselves as we assume they are
  // immutable (today).
  iree_slim_mutex_t mutex;

  // Total capacity of the entries list in elements.
  iree_host_size_t entry_capacity;
  // Currently used entry count in elements.
  iree_host_size_t entry_count;
  // Dense list of entries in the index. Grows as needed.
  iree_io_parameter_index_entry_t** entries;
};

IREE_API_EXPORT iree_status_t iree_io_parameter_index_create(
    iree_allocator_t host_allocator, iree_io_parameter_index_t** out_index) {
  IREE_ASSERT_ARGUMENT(out_index);
  *out_index = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_io_parameter_index_t* index = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_allocator_malloc(host_allocator, sizeof(*index), (void**)&index));
  iree_atomic_ref_count_init(&index->ref_count);
  index->host_allocator = host_allocator;

  iree_slim_mutex_initialize(&index->mutex);

  // Grown on first use. We could allocate a bit of inline storage or take an
  // optional initial capacity for callers that know.
  index->entry_capacity = 0;
  index->entry_count = 0;
  index->entries = NULL;

  *out_index = index;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_io_parameter_index_destroy(iree_io_parameter_index_t* index) {
  IREE_ASSERT_ARGUMENT(index);
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_allocator_t host_allocator = index->host_allocator;

  for (iree_host_size_t i = 0; i < index->entry_count; ++i) {
    iree_io_parameter_index_entry_t* entry = index->entries[i];
    switch (entry->type) {
      default:
        break;
      case IREE_IO_PARAMETER_INDEX_ENTRY_STORAGE_TYPE_FILE:
        iree_io_file_handle_release(entry->storage.file.handle);
        break;
    }
    iree_allocator_free(host_allocator, entry);
  }
  if (index->entries) {
    iree_allocator_free(host_allocator, index->entries);
  }

  iree_slim_mutex_deinitialize(&index->mutex);

  iree_allocator_free(host_allocator, index);

  IREE_TRACE_ZONE_END(z0);
}

IREE_API_EXPORT void iree_io_parameter_index_retain(
    iree_io_parameter_index_t* index) {
  if (IREE_LIKELY(index)) {
    iree_atomic_ref_count_inc(&index->ref_count);
  }
}

IREE_API_EXPORT void iree_io_parameter_index_release(
    iree_io_parameter_index_t* index) {
  if (IREE_LIKELY(index) && iree_atomic_ref_count_dec(&index->ref_count) == 1) {
    iree_io_parameter_index_destroy(index);
  }
}

IREE_API_EXPORT iree_host_size_t
iree_io_parameter_index_count(iree_io_parameter_index_t* index) {
  IREE_ASSERT_ARGUMENT(index);
  iree_slim_mutex_lock(&index->mutex);
  iree_host_size_t count = index->entry_count;
  iree_slim_mutex_unlock(&index->mutex);
  return count;
}

static iree_status_t iree_io_parameter_index_reserve_unsafe(
    iree_io_parameter_index_t* index, iree_host_size_t new_capacity) {
  IREE_ASSERT_ARGUMENT(index);
  if (new_capacity < index->entry_capacity) return iree_ok_status();
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, new_capacity);

  iree_io_parameter_index_entry_t** new_entries = index->entries;
  iree_status_t status = iree_allocator_realloc(
      index->host_allocator, new_capacity * sizeof(index->entries[0]),
      (void**)&new_entries);
  if (iree_status_is_ok(status)) {
    index->entry_capacity = new_capacity;
    index->entries = new_entries;
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_io_parameter_index_reserve(
    iree_io_parameter_index_t* index, iree_host_size_t new_capacity) {
  IREE_ASSERT_ARGUMENT(index);
  iree_slim_mutex_lock(&index->mutex);
  iree_status_t status =
      iree_io_parameter_index_reserve_unsafe(index, new_capacity);
  iree_slim_mutex_unlock(&index->mutex);
  return status;
}

IREE_API_EXPORT iree_status_t
iree_io_parameter_index_add(iree_io_parameter_index_t* index,
                            const iree_io_parameter_index_entry_t* entry) {
  IREE_ASSERT_ARGUMENT(index);
  IREE_ASSERT_ARGUMENT(entry);
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_TEXT(z0, entry->key.data, entry->key.size);
  iree_slim_mutex_lock(&index->mutex);

  // Grow the index if needed (double each time after some initial minimum).
  iree_status_t status = iree_ok_status();
  if (index->entry_count == index->entry_capacity) {
    status = iree_io_parameter_index_reserve_unsafe(
        index, iree_max(16, index->entry_capacity * 2));
  }

  // Clone the entry memory. We allocate it as a single slab and stash the
  // pointers for easier access by callers. Entries themselves are never
  // reallocated so the pointers are safe to embed.
  iree_io_parameter_index_entry_t* cloned_entry = NULL;
  if (iree_status_is_ok(status)) {
    iree_host_size_t total_size =
        sizeof(*cloned_entry) + entry->key.size + entry->metadata.data_length;
    status = iree_allocator_malloc(index->host_allocator, total_size,
                                   (void**)&cloned_entry);
  }
  if (iree_status_is_ok(status)) {
    cloned_entry->key = iree_make_string_view(
        (char*)cloned_entry + sizeof(*cloned_entry), entry->key.size);
    cloned_entry->metadata =
        iree_const_byte_span_is_empty(entry->metadata)
            ? iree_const_byte_span_empty()
            : iree_make_const_byte_span(
                  (uint8_t*)cloned_entry->key.data + cloned_entry->key.size,
                  entry->metadata.data_length);
    cloned_entry->length = entry->length;
    cloned_entry->type = entry->type;
    switch (entry->type) {
      default:
        break;
      case IREE_IO_PARAMETER_INDEX_ENTRY_STORAGE_TYPE_SPLAT:
        cloned_entry->storage.splat = entry->storage.splat;
        break;
      case IREE_IO_PARAMETER_INDEX_ENTRY_STORAGE_TYPE_FILE:
        cloned_entry->storage.file = entry->storage.file;
        iree_io_file_handle_retain(cloned_entry->storage.file.handle);
        break;
    }
    memcpy((void*)cloned_entry->key.data, entry->key.data, entry->key.size);
    memcpy((void*)cloned_entry->metadata.data, entry->metadata.data,
           entry->metadata.data_length);

    // Append the entry to the file index.
    index->entries[index->entry_count++] = cloned_entry;
  }

  iree_slim_mutex_unlock(&index->mutex);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_io_parameter_index_get(
    iree_io_parameter_index_t* index, iree_host_size_t i,
    const iree_io_parameter_index_entry_t** out_entry) {
  IREE_ASSERT_ARGUMENT(index);
  IREE_ASSERT_ARGUMENT(out_entry);
  *out_entry = NULL;
  iree_slim_mutex_lock(&index->mutex);

  iree_status_t status = iree_ok_status();
  if (i < index->entry_count) {
    *out_entry = index->entries[i];
  } else {
    status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "entry %" PRIhsz " out of range (have %" PRIhsz
                              " entries in the index)",
                              i, index->entry_count);
  }

  iree_slim_mutex_unlock(&index->mutex);
  return status;
}

IREE_API_EXPORT iree_status_t iree_io_parameter_index_lookup(
    iree_io_parameter_index_t* index, iree_string_view_t key,
    const iree_io_parameter_index_entry_t** out_entry) {
  IREE_ASSERT_ARGUMENT(index);
  IREE_ASSERT_ARGUMENT(out_entry);
  *out_entry = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_TEXT(z0, key.data, key.size);
  iree_slim_mutex_lock(&index->mutex);

  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < index->entry_count; ++i) {
    const iree_io_parameter_index_entry_t* entry = index->entries[i];
    if (iree_string_view_equal(key, entry->key)) {
      *out_entry = entry;
      break;
    }
  }
  if (*out_entry == NULL) {
    status = iree_make_status(IREE_STATUS_NOT_FOUND,
                              "no parameter found in index with key '%.*s'",
                              (int)key.size, key.data);
  }

  iree_slim_mutex_unlock(&index->mutex);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_io_parameter_index_dump(
    iree_string_view_t scope, iree_io_parameter_index_t* index,
    iree_string_builder_t* builder) {
  iree_host_size_t entry_count = iree_io_parameter_index_count(index);
  uint64_t total_bytes = 0;
  for (iree_host_size_t i = 0; i < entry_count; ++i) {
    const iree_io_parameter_index_entry_t* entry = NULL;
    IREE_RETURN_IF_ERROR(iree_io_parameter_index_get(index, i, &entry));
    total_bytes += entry->length;
  }
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(
      builder,
      "//"
      "===-----------------------------------------------------------------"
      "---------------------------------------------===//\n"));
  IREE_RETURN_IF_ERROR(
      iree_string_builder_append_cstring(builder, "// Parameter scope "));
  if (iree_string_view_is_empty(scope)) {
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_cstring(builder, "<global>"));
  } else {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, "`%.*s`", (int)scope.size, scope.data));
  }
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder, " (%" PRIhsz " entries, %" PRIu64 " total bytes)\n", entry_count,
      total_bytes));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(
      builder,
      "//===------------+------------------+------------------+------------"
      "-----------------------------------------------===//\n"));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(
      builder,
      "//         Start |              End |           Length | Key\n"));
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(
      builder,
      "//---------------+------------------+------------------+------------"
      "--------------------------------------------------//\n"));
  for (iree_host_size_t i = 0; i < entry_count; ++i) {
    const iree_io_parameter_index_entry_t* entry = NULL;
    IREE_RETURN_IF_ERROR(iree_io_parameter_index_get(index, i, &entry));
    switch (entry->type) {
      case IREE_IO_PARAMETER_INDEX_ENTRY_STORAGE_TYPE_SPLAT: {
        IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
            builder,
            "               - |                - | %16" PRIu64 " | `%.*s`\n",
            entry->length, (int)entry->key.size, entry->key.data));
        break;
      }
      case IREE_IO_PARAMETER_INDEX_ENTRY_STORAGE_TYPE_FILE: {
        IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
            builder, "%16" PRIu64 " | %16" PRIu64 " | %16" PRIu64 " | `%.*s`\n",
            entry->storage.file.offset,
            entry->storage.file.offset + entry->length, entry->length,
            (int)entry->key.size, entry->key.data));
        break;
      }
      default: {
        IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
            builder,
            "               ? |               ? | %16" PRIu64 " | `%.*s`\n",
            entry->length, (int)entry->key.size, entry->key.data));
        break;
      }
    }
  }
  IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "\n"));
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_io_parameter_index_fprint(
    FILE* file, iree_string_view_t scope, iree_io_parameter_index_t* index) {
  iree_string_builder_t builder;
  iree_string_builder_initialize(index->host_allocator, &builder);
  iree_status_t status = iree_io_parameter_index_dump(scope, index, &builder);
  if (iree_status_is_ok(status)) {
    fprintf(file, "%.*s", (int)iree_string_builder_size(&builder),
            iree_string_builder_buffer(&builder));
  }
  iree_string_builder_deinitialize(&builder);
  return status;
}
