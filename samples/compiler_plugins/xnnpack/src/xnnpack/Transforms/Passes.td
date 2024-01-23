// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include "mlir/Pass/PassBase.td"

#ifndef IREE_XNNPACK_TRANSFORMS_PASSES
#define IREE_XNNPACK_TRANSFORMS_PASSES

def LegalizeXnnpack : Pass<"iree-xnnpack-legalize", "mlir::ModuleOp"> {
  let summary = "Legalizes the xnnpack ops";
}

#endif // IREE_XNNPACK_TRANSFORMS_PASSES
