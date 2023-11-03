// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "xnnpack_sample/IR/XnnpackOps.h"
#include "xnnpack_sample/Transforms/Passes.h"

#define GEN_PASS_DEF_LEGALIZEXNNPACK
#include "xnnpack_sample/Transforms/Passes.h.inc"

namespace mlir::iree_compiler::IREE::Xnnpack {
namespace {

class LegalizeXnnpackPass
    : public ::impl::LegalizeXnnpackBase<LegalizeXnnpackPass> {
 public:
  void runOnOperation() override {
    auto *context = &getContext();
    // TODO: This is all just a placeholder. To make it real, we should be
    // checking if the import already exists and likely doing some more fancy
    // lowering.
    // Add imports.
    auto m = getOperation();
    auto importBuilder = OpBuilder::atBlockBegin(m.getBody());
    importBuilder
        .create<func::FuncOp>(m.getLoc(), "xnnpack.print",
                              FunctionType::get(context, {}, {}))
        .setPrivate();

    // Legalize operations.
    m.walk([&](Operation *op) {
      if (auto printOp = dyn_cast<IREE::Xnnpack::PrintOp>(op)) {
        OpBuilder b(op);
        b.create<func::CallOp>(printOp.getLoc(), "xnnpack.print", TypeRange{});
        printOp.erase();
      }
    });
  }
};

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> createLegalizeXnnpackPass() {
  return std::make_unique<LegalizeXnnpackPass>();
}

}  // namespace mlir::iree_compiler::IREE::Xnnpack
