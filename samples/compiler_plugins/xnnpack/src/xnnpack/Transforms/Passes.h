// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_SAMPLES_COMPILER_PLUGINS_XNNPACK_TRANSFORMS_PASSES_H_
#define IREE_SAMPLES_COMPILER_PLUGINS_XNNPACK_TRANSFORMS_PASSES_H_

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

namespace mlir::iree_compiler::IREE::Xnnpack {
#define GEN_PASS_DECL
#include "xnnpack/Transforms/Passes.h.inc"

void registerXnnpackPluginTransformsPasses();

}  // namespace mlir::iree_compiler::IREE::Xnnpack

#endif  // IREE_SAMPLES_COMPILER_PLUGINS_XNNPACK_TRANSFORMS_PASSES_H_
