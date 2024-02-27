// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Visitors.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "xnnpack/IR/XnnpackOps.h"
#include "xnnpack/Transforms/Passes.h"

namespace mlir::iree_compiler::IREE::Xnnpack {
#define GEN_PASS_DEF_SETIDATTRIBUTES
#include "xnnpack/Transforms/Passes.h.inc"

namespace {
// Checks if `val` is a constant by going up the use-def tree to see if the
// leaves of the tree are `stablehlo` constants.
//
// Note: a returned value of false means that this function was unable to
// determine if the value is constant or not.
static bool isConstant(Value val) {
  SmallVector<Value> stack{val};
  SmallVector<Operation *> zeroOperandOps;
  while (!stack.empty()) {
    Value currVal = stack.pop_back_val();
    if (auto defOp = currVal.getDefiningOp()) {
      if (defOp->getNumOperands() == 0) zeroOperandOps.push_back(defOp);
      stack.append(SmallVector<Value>(defOp->getOperands()));
    } else {
      return false;
    }
  }
  return llvm::all_of(zeroOperandOps, [](Operation *op) {
    return isa<stablehlo::ConstantOp>(op);
  });
}

class SetIdAttributesPass
    : public impl::SetIdAttributesBase<SetIdAttributesPass> {
 public:
  SetIdAttributesPass() = default;
  SetIdAttributesPass(const SetIdAttributesPass &) {}
  void runOnOperation() override {
    auto m = getOperation();
    DenseMap<Value, int64_t> kernelIds;
    auto result =
        m.walk([&](Xnnpack::FullyConnectedNcQd8F32Qc4wOp fc) -> WalkResult {
          Value kernel = fc.getKernel();
          if (!isConstant(kernel))
            return fc.emitError(
                "unimplemented: found kernel that is not guaranteed to be "
                "constant");
          if (!kernelIds.contains(kernel)) kernelIds[kernel] = kernelIds.size();
          auto kernelIdAttr = IntegerAttr::get(IndexType::get(m.getContext()),
                                               kernelIds[kernel]);
          fc.setKernelIdAttr(kernelIdAttr);
          return WalkResult::advance();
        });

    if (result.wasInterrupted()) return signalPassFailure();
  }
};

}  // namespace
}  // namespace mlir::iree_compiler::IREE::Xnnpack
