// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "xnnpack_sample/Conversion/Passes.h"

#define GEN_PASS_DEF_CONVERTSTABLEHLOTOXNNPACK
#include "xnnpack_sample/Conversion/Passes.h.inc"

namespace mlir::iree_compiler::IREE::Xnnpack {
namespace {
class ConvertStablehloToXnnpackPass
    : public ::impl::ConvertStablehloToXnnpackBase<
          ConvertStablehloToXnnpackPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {}
  void runOnOperation() override {}
};

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> createConvertStablehloToXnnpackPass() {
  return std::make_unique<ConvertStablehloToXnnpackPass>();
}

}  // namespace mlir::iree_compiler::IREE::Xnnpack
