// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Util/Analysis/Constant/OpOracle.h"

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir::iree_compiler::IREE::Util {

namespace {

void populateEscapingProducers(Operation *parentOp, ConstExprOpInfo &info) {
  SmallPtrSet<Operation *, 8> containedOps;
  parentOp->walk<WalkOrder::PreOrder>([&](Operation *itOp) {
    containedOps.insert(parentOp);
    // For the outer-most op, consider that all operands escape.
    if (itOp == parentOp) {
      info.producers.insert(itOp->getOperands().begin(),
                            itOp->getOperands().end());
      return;
    }

    // For nested operations, only consider that they escape if they are
    // defined outside of the parent.
    for (Value operand : itOp->getOperands()) {
      Block *block = operand.getParentBlock();
      if (!containedOps.contains(block->getParentOp())) {
        info.producers.insert(operand);
      }
    }
  });
}

// Enforce a limited allow-list of types that are legal to consider
// constexpr operand or result types. Given MLIR's open type system,
// it is best to be conservative here, and we limit to known value
// types.
bool isLegalConstExprType(Type t) {
  // If implementing the hoistable interface, just return what the interface
  // says.
  if (auto hoistableType = dyn_cast<IREE::Util::HoistableTypeInterface>(t)) {
    return hoistableType.isHoistableType();
  }
  if (llvm::isa<IntegerType, FloatType>(t)) {
    // TODO: We shouldn't need to be this conservative about the bit widths we
    // support, but for now the consteval JIT has interop limitations. Lift
    // this restriction when the JIT interops for all types.
    auto bitWidth = t.getIntOrFloatBitWidth();
    return bitWidth == 1 || bitWidth == 8 || bitWidth == 16 || bitWidth == 32 ||
           bitWidth == 64;
  }

  if (llvm::isa<IndexType>(t)) {
    return true;
  }

  if (auto tensorType = llvm::dyn_cast<TensorType>(t)) {
    return isLegalConstExprType(tensorType.getElementType());
  }

  return false;
}

// Check if the op can be an eligible const expr.
bool isEligibleConstExpr(Operation *op) {
  // We have a specific allow-list for Linalg ops because we want to consider
  // new additions carefully.
  if (op->getDialect() ==
      op->getContext()->getOrLoadDialect<linalg::LinalgDialect>()) {
    // Structured op implementations and a handful of pure ops are included.
    // Notably: IndexOp is not included because it establishes a hidden
    // dependency to the iterator and is non-const.
    if (llvm::isa<linalg::LinalgOp>(op) || llvm::isa<tensor::PadOp>(op) ||
        llvm::isa<tensor::EmptyOp>(op)) {
      return true;
    }
    return false;
  }

  // Target-dependent ops are not const-expr.
  // TODO(#14887): Use trait/interface instead.
  if (isa<IREE::LinalgExt::UpperBoundTileSizeOp,
          IREE::LinalgExt::SetEncodingOp>(op)) {
    return false;
  }

  // By default, ops without results are not const-expr.
  if (op->getNumResults() == 0) {
    return false;
  }

  // Forbid if illegal result types. It is sufficient to verify result
  // types since all constexpr values must come from a result somewhere
  // in the analyzed tree.
  if (!llvm::all_of(op->getResultTypes(), isLegalConstExprType)) {
    return false;
  }

  // Forbid if part of a parent that should be treated atomically.
  if (op->getParentOfType<linalg::LinalgOp>()) {
    return false;
  }

  // Optimization barriers cannot be folded.
  if (isa<IREE::Util::OptimizationBarrierOp>(op)) {
    return false;
  }

  // Special carve-out for unregistered testing ops.
  // Since these are unregistered, we have to make this carve-out
  // after any verification on structure but before any verification
  // of traits (like effects). These specific unregistered ops
  // override default traits.
  if (!op->isRegistered()) {
    // Reject.
    if (op->getName().getStringRef() == "iree_unregistered.var_expr") {
      return false;
    }
    // Accept.
    if (op->getName().getStringRef() ==
            "iree_unregistered.non_leaf_const_expr" ||
        op->getName().getStringRef() == "iree_unregistered.const_expr") {
      return true;
    }
  }

  // By default any effects make it non const-expr.
  if (!isMemoryEffectFree(op)) {
    return false;
  }

  return true;
}

} // namespace

void registerConstExprDependentDialects(DialectRegistry &registry) {
  registry.insert<IREE::Util::UtilDialect>();
  registry.insert<linalg::LinalgDialect>();
}

bool isLegalConstExprRootType(Type t) { return isLegalConstExprType(t); }

ConstExprOpInfo ConstExprOpInfo::getForOp(Operation *op) {
  ConstExprOpInfo info;
  info.isEligible = isEligibleConstExpr(op);
  // Populate the producers for both eligible and ineligible cases, as we need
  // the producers of ineligible op to identify hoistable constant producers.
  populateEscapingProducers(op, info);
  return info;
}

bool isHoistableConstExprLeaf(const ConstExprAnalysis::ConstValueInfo *info) {
  Operation *op = info->getOperation();
  if (!op->isRegistered()) {
    if (op->getName().getStringRef() ==
        "iree_unregistered.non_leaf_const_expr") {
      return false;
    }
  }

  // First check whether we should hoist this kind of operation. Type local
  // decisions should always come last.

  // Generally, we prefer to not hoist broadcasts.
  if (auto genericOp = dyn_cast<linalg::GenericOp>(op)) {
    // Detect op that only broadcast input as fusing them makes the new
    // op cheaper.
    if (genericOp.getNumParallelLoops() == genericOp.getNumLoops() &&
        isa<linalg::YieldOp>(genericOp.getBody()->front())) {
      for (OpOperand *opOperand : genericOp.getDpsInputOperands()) {
        AffineMap indexingMap = genericOp.getMatchingIndexingMap(opOperand);
        if (indexingMap.isProjectedPermutation() &&
            indexingMap.getNumDims() != indexingMap.getNumResults()) {
          return false;
        }
      }
    }
  }

  // Never hoist empty. These are sometimes used for pure shape metadata
  // and must not be separated from their consumers.
  if (isa<tensor::EmptyOp, tensor::ExpandShapeOp, tensor::CollapseShapeOp>(
          op)) {
    return false;
  }

  // If implementing the HoistableTypeInterface, just return what the interface
  // says.
  if (auto hoistableType = dyn_cast<IREE::Util::HoistableTypeInterface>(
          info->constValue.getType())) {
    return hoistableType.isHoistableLeafType();
  }

  // Never hoist sub-byte aligned values: in legal programs, these will be
  // cast or packed in some successor.
  if (auto integerType = llvm::dyn_cast<IntegerType>(
          getElementTypeOrSelf(info->constValue.getType()))) {
    if (integerType.getWidth() % 8 != 0) {
      return false;
    }
  }

  return true;
}

bool isHoistableConstExprConsumingOperand(OpOperand *operand) {
  Operation *op = operand->getOwner();
  // For linalg ops, we only want to hoist inputs.
  if (auto structuredOp = dyn_cast<linalg::LinalgOp>(op)) {
    return operand->getOperandNumber() < structuredOp.getNumDpsInputs();
  }

  // Fallback to yes.
  return true;
}

} // namespace mlir::iree_compiler::IREE::Util
