// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/IREECodegenDialect.h"
#include "iree/compiler/Codegen/Dialect/UKernelOps.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "xnnpack_sample/IR/XnnpackOps.h"
#include "xnnpack_sample/Transforms/Passes.h"

#define GEN_PASS_DEF_LEGALIZEXNNPACK
#include "xnnpack_sample/Transforms/Passes.h.inc"

namespace mlir::iree_compiler::IREE::Xnnpack {
namespace {
static FailureOr<func::FuncOp> createFuncOp(
    RewriterBase &rewriter, Location loc, FunctionType type,
    llvm::StringRef name,
    function_ref<LogicalResult(RewriterBase &, Location,
                               ArrayRef<BlockArgument>, ArrayRef<Type>)>
        bodyBuild) {
  auto func = rewriter.create<func::FuncOp>(loc, name, type);
  auto *entryBlock = func.addEntryBlock();
  auto entryBuilder = OpBuilder::atBlockBegin(entryBlock);
  IRRewriter entryRewriter(entryBuilder);
  if (failed(bodyBuild(entryRewriter, loc, entryBlock->getArguments(),
                       type.getResults()))) {
    return rewriter.notifyMatchFailure(func,
                                       "unable to create body of function");
  }
  return func;
}

static FailureOr<func::FuncOp> createUKernelGeneric(
    RewriterBase &moduleRewriter, Operation *op) {
  auto funcType = FunctionType::get(op->getContext(), op->getOperandTypes(),
                                    op->getResultTypes());
  llvm::StringRef opName = op->getName().getStringRef();
  auto func = createFuncOp(
      moduleRewriter, op->getLoc(), funcType, opName,
      [opName, op](RewriterBase &rewriter, Location loc,
                   ArrayRef<BlockArgument> operands,
                   ArrayRef<Type> resultTypes) -> LogicalResult {
        Value firstOperand = operands[0];
        // TODO(ramiro050): check that operands are tensors
        RankedTensorType operandType =
            firstOperand.getType().cast<RankedTensorType>();
        Type elementType = operandType.getElementType();
        if (operandType.getRank() != 1) {
          return rewriter.notifyMatchFailure(op, "unimplemented: rank != 1");
        }
        if (resultTypes.size() != 1) {
          return rewriter.notifyMatchFailure(op,
                                             "unimplemented: multiple returns");
        }
        Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
        Value d0 = rewriter.create<tensor::DimOp>(loc, firstOperand, c0);
        Value dest = rewriter.create<tensor::EmptyOp>(
            loc, getAsOpFoldResult({d0}), elementType);

        auto dispatchRegion = rewriter.create<IREE::Flow::DispatchRegionOp>(
            loc, resultTypes, /*result_dims=*/ValueRange{d0},
            /*workload=*/ValueRange{});
        Block &dispatchBody = dispatchRegion.getBody().emplaceBlock();
        {
          OpBuilder::InsertionGuard guard(rewriter);
          rewriter.setInsertionPointToStart(&dispatchBody);
          auto ukernel =
              rewriter
                  .create<IREE::Codegen::UKernelGenericOp>(
                      loc, resultTypes, (opName + "_workgroup").str(), operands,
                      dest, ValueRange{d0}, /*fn_def_attrs=*/nullptr,
                      /*strided_outer_dims=*/nullptr)
                  .getResults();
          rewriter.create<Flow::ReturnOp>(loc, ukernel);
        }

        Block &dispatchWorkgroupCount =
            dispatchRegion.getWorkgroupCount().emplaceBlock();
        {
          OpBuilder::InsertionGuard guard(rewriter);
          rewriter.setInsertionPointToStart(&dispatchWorkgroupCount);
          Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
          rewriter.create<Flow::ReturnOp>(loc, ValueRange{c1, c1, c1});
        }

        rewriter.create<func::ReturnOp>(loc, dispatchRegion.getResults());
        return success();
      });
  return func;
}

class LegalizeXnnpackPass
    : public ::impl::LegalizeXnnpackBase<LegalizeXnnpackPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<tensor::TensorDialect, IREE::Flow::FlowDialect,
                    IREE::Codegen::IREECodegenDialect>();
  }
  void runOnOperation() override {
    auto m = getOperation();
    auto importBuilder = OpBuilder::atBlockBegin(m.getBody());
    IRRewriter rewriter(importBuilder);
    auto result = m.walk([&](Operation *op) -> WalkResult {
      if (op->getDialect()->getNamespace() != "xnnpack")
        return WalkResult::advance();

      FailureOr<func::FuncOp> ukernelGeneric =
          createUKernelGeneric(rewriter, op);
      if (failed(ukernelGeneric)) return WalkResult::interrupt();
      ukernelGeneric->setPrivate();

      OpBuilder b(op);
      auto func =
          b.create<func::CallOp>(op->getLoc(), ukernelGeneric->getName(),
                                 op->getResultTypes(), op->getOperands());
      op->replaceAllUsesWith(func.getResults());
      op->erase();
      return WalkResult::advance();
    });
    if (result.wasInterrupted()) signalPassFailure();
  }
};

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> createLegalizeXnnpackPass() {
  return std::make_unique<LegalizeXnnpackPass>();
}

}  // namespace mlir::iree_compiler::IREE::Xnnpack
