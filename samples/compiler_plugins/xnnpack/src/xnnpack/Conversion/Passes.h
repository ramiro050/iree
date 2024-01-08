// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_SAMPLES_COMPILER_PLUGINS_XNNPACK_CONVERSION_PASSES_H_
#define IREE_SAMPLES_COMPILER_PLUGINS_XNNPACK_CONVERSION_PASSES_H_

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

namespace mlir::iree_compiler::IREE::Xnnpack {

std::unique_ptr<OperationPass<ModuleOp>> createConvertStablehloToXnnpackPass();

void registerXnnpackPluginConversionPasses();

}  // namespace mlir::iree_compiler::IREE::Xnnpack

#endif  // IREE_SAMPLES_COMPILER_PLUGINS_XNNPACK_CONVERSION_PASSES_H_
