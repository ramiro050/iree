// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMCPU/PassDetail.h"
#include "iree/compiler/Codegen/LLVMCPU/Passes.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler {

namespace {
struct LLVMCPUEmitVectorizationRemarksPass
    : LLVMCPUEmitVectorizationRemarksBase<LLVMCPUEmitVectorizationRemarksPass> {
  void runOnOperation() override;
};
} // namespace

void LLVMCPUEmitVectorizationRemarksPass::runOnOperation() {
  auto funcOp = getOperation();
  bool dump = false;
  funcOp.walk([&](linalg::LinalgOp op) {
    op.emitWarning("op is not vectorized");
    dump = true;
  });
  if (dump) {
    funcOp.emitWarning("found one or more ops not vectorized");
  }
}

std::unique_ptr<OperationPass<func::FuncOp>>
createLLVMCPUEmitVectorizationRemarksPass() {
  return std::make_unique<LLVMCPUEmitVectorizationRemarksPass>();
}

} // namespace mlir::iree_compiler
