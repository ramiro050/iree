// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "xnnpack/Conversion/Passes.h"

namespace mlir::iree_compiler::IREE::Xnnpack {
namespace {
#define GEN_PASS_REGISTRATION
#include "xnnpack/Conversion/Passes.h.inc"
}  // namespace

void registerXnnpackPluginConversionPasses() {
  // Generated.
  registerPasses();
}

}  // namespace mlir::iree_compiler::IREE::Xnnpack