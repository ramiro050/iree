// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include "mlir/Pass/PassBase.td"

#ifndef IREE_XNNPACK_CONVERSION_PASSES
#define IREE_XNNPACK_CONVERSION_PASSES

def ConvertStablehloToXnnpack : Pass<"iree-stablehlo-to-xnnpack", "mlir::ModuleOp"> {
  let summary = "Convert StableHLO ops to XNNPACK dialect ops";
  let constructor = [{
    ::mlir::iree_compiler::IREE::Xnnpack::createConvertStablehloToXnnpackPass()
  }];
}

#endif // IREE_XNNPACK_CONVERSION_PASSES
