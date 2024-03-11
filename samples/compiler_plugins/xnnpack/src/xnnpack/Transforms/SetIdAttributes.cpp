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
// Checks if the kernel of `fc` is a constant.
//
// Note: a returned value of false means that this function was unable to
// determine if the value is constant or not.
static bool kernelIsConstant(FullyConnectedNcQd8F32Qc4wOp fc) {
  Value kernel = fc.getKernel();
  // The current implementation of the fully connected XNNPACK op requires that
  // we transform the kernel from signed int space to unsigned int space. This
  // is currently done through an XOR operation, hence the special case handling
  // here.
  if (auto xorOp = kernel.getDefiningOp<stablehlo::XorOp>())
    kernel = xorOp.getLhs();
  return kernel.getDefiningOp<stablehlo::ConstantOp>();
}

class SetIdAttributesPass
    : public impl::SetIdAttributesBase<SetIdAttributesPass> {
 public:
  SetIdAttributesPass() = default;
  SetIdAttributesPass(const SetIdAttributesPass &) {}
  void runOnOperation() override {
    auto m = getOperation();
    DenseMap<Value, int64_t> kernelIds;
    auto result = m.walk([&](FullyConnectedNcQd8F32Qc4wOp fc) -> WalkResult {
      if (!kernelIsConstant(fc))
        return fc.emitError(
            "unimplemented: found kernel that is not guaranteed to be "
            "constant");
      Value kernel = fc.getKernel();
      if (!kernelIds.contains(kernel)) kernelIds[kernel] = kernelIds.size();
      auto kernelIdAttr =
          IntegerAttr::get(IndexType::get(m.getContext()), kernelIds[kernel]);
      fc.setKernelIdAttr(kernelIdAttr);
      return WalkResult::advance();
    });

    if (result.wasInterrupted()) return signalPassFailure();
  }
};

}  // namespace
}  // namespace mlir::iree_compiler::IREE::Xnnpack
