// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_LINALGEXT_TRANSFORMS_PASS_DETAIL_H_
#define IREE_COMPILER_DIALECT_LINALGEXT_TRANSFORMS_PASS_DETAIL_H_

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::IREE::LinalgExt {

#define GEN_PASS_CLASSES

#include "iree/compiler/Dialect/LinalgExt/Transforms/Passes.h.inc" // IWYU pragma: keep

} // namespace mlir::iree_compiler::IREE::LinalgExt

#endif // IREE_COMPILER_DIALECT_LINALGEXT_TRANSFORMS_PASS_DETAIL_H_
