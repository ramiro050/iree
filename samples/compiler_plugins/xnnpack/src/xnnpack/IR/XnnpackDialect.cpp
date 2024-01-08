// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "xnnpack/IR/XnnpackDialect.h"

#include "xnnpack/IR/XnnpackOps.h"

namespace mlir::iree_compiler::IREE::Xnnpack {

void XnnpackDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "xnnpack/IR/XnnpackOps.cpp.inc"
      >();
}

}  // namespace mlir::iree_compiler::IREE::Xnnpack

#include "xnnpack/IR/XnnpackDialect.cpp.inc"
