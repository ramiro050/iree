// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "xnnpack/IR/XnnpackOps.h"

#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"

namespace mlir::iree_compiler::IREE::Xnnpack {

LogicalResult BatchMatrixMultiplyOp::verify() {
  auto aType = getA().getType().cast<RankedTensorType>();
  auto bType = getB().getType().cast<RankedTensorType>();
  if (aType.getRank() < 3 || bType.getRank() < 3) {
    return emitOpError()
           << "expected operands to have rank >= 3, but got ranks: "
           << aType.getRank() << " and " << bType.getRank();
  }

  if (aType.getRank() != bType.getRank()) {
    return emitOpError()
           << "expected operands to have the same rank, but got ranks: "
           << aType.getRank() << " and " << bType.getRank();
  }

  ArrayRef<int64_t> aShape = aType.getShape();
  ArrayRef<int64_t> bShape = bType.getShape();
  for (auto [aDim, bDim] :
       llvm::zip(aShape.drop_back(2), bShape.drop_back(2))) {
    if (aDim != ShapedType::kDynamic && bDim != ShapedType::kDynamic &&
        aDim != bDim) {
      return emitOpError()
             << "expected first N-2 dimensions to match, but dimension sizes "
             << aDim << " != " << bDim;
    }
  }

  int64_t aKDim = aShape.back();
  int64_t bKDim = bShape.drop_back().back();
  if (aKDim != ShapedType::kDynamic && bKDim != ShapedType::kDynamic &&
      aKDim != bKDim) {
    return emitOpError() << "expected reduction dimension to be the same in "
                            "both operands, but got "
                         << aKDim << " and " << bKDim;
  }
  return success();
}
}  // namespace mlir::iree_compiler::IREE::Xnnpack

//===----------------------------------------------------------------------===//
// TableGen definitions (intentionally last)
//===----------------------------------------------------------------------===//
// clang-format off
#define GET_OP_CLASSES
#include "xnnpack/IR/XnnpackOps.cpp.inc" // IWYU pragma: keep
// clang-format on
