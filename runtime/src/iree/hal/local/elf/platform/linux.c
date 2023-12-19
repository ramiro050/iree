// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/local/elf/platform.h"

#if defined(IREE_PLATFORM_ANDROID) || defined(IREE_PLATFORM_LINUX)

#include <errno.h>
#include <sys/mman.h>
#include <unistd.h>

//==============================================================================
// Virtual address space manipulation
//==============================================================================

static int iree_memory_access_to_prot(iree_memory_access_t access) {
  int prot = 0;
  if (access & IREE_MEMORY_ACCESS_READ) prot |= PROT_READ;
  if (access & IREE_MEMORY_ACCESS_WRITE) prot |= PROT_WRITE;
  if (access & IREE_MEMORY_ACCESS_EXECUTE) prot |= PROT_EXEC;
  return prot;
}

iree_status_t iree_memory_view_reserve(iree_memory_view_flags_t flags,
                                       iree_host_size_t total_length,
                                       iree_allocator_t host_allocator,
                                       void** out_base_address) {
  *out_base_address = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  int mmap_prot = PROT_NONE;
  int mmap_flags = MAP_PRIVATE | MAP_ANON | MAP_NORESERVE;

  iree_status_t status = iree_ok_status();
  void* base_address = mmap(NULL, total_length, mmap_prot, mmap_flags, -1, 0);
  if (base_address == MAP_FAILED) {
    status = iree_make_status(iree_status_code_from_errno(errno),
                              "mmap reservation failed");
  }

  *out_base_address = base_address;
  IREE_TRACE_ZONE_END(z0);
  return status;
}

void iree_memory_view_release(void* base_address, iree_host_size_t total_length,
                              iree_allocator_t host_allocator) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // NOTE: return value ignored as this is a shutdown path.
  munmap(base_address, total_length);

  IREE_TRACE_ZONE_END(z0);
}

iree_status_t iree_memory_view_commit_ranges(
    void* base_address, iree_host_size_t range_count,
    const iree_byte_range_t* ranges, iree_memory_access_t initial_access) {
  IREE_TRACE_ZONE_BEGIN(z0);

  int mmap_prot = iree_memory_access_to_prot(initial_access);
  int mmap_flags = MAP_PRIVATE | MAP_ANON | MAP_FIXED;

  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < range_count; ++i) {
    void* range_start = NULL;
    iree_host_size_t aligned_length = 0;
    iree_page_align_range(base_address, ranges[i], getpagesize(), &range_start,
                          &aligned_length);
    void* result =
        mmap(range_start, aligned_length, mmap_prot, mmap_flags, -1, 0);
    if (result == MAP_FAILED) {
      status = iree_make_status(iree_status_code_from_errno(errno),
                                "mmap commit failed");
      break;
    }
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_memory_view_protect_ranges(void* base_address,
                                              iree_host_size_t range_count,
                                              const iree_byte_range_t* ranges,
                                              iree_memory_access_t new_access) {
  IREE_TRACE_ZONE_BEGIN(z0);

  int mmap_prot = iree_memory_access_to_prot(new_access);

  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < range_count; ++i) {
    void* range_start = NULL;
    iree_host_size_t aligned_length = 0;
    iree_page_align_range(base_address, ranges[i], getpagesize(), &range_start,
                          &aligned_length);
    int ret = mprotect(range_start, aligned_length, mmap_prot);
    if (ret != 0) {
      status = iree_make_status(iree_status_code_from_errno(errno),
                                "mprotect failed");
      break;
    }
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

#endif  // IREE_PLATFORM_*
