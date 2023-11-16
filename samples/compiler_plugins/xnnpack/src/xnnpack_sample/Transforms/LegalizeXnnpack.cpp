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
static func::FuncOp createFuncOp(
    OpBuilder &b, Location loc, FunctionType type, llvm::StringRef name,
    function_ref<void(OpBuilder &, Location, ArrayRef<BlockArgument>,
                      ArrayRef<Type>)>
        bodyBuild) {
  auto func = b.create<func::FuncOp>(loc, name, type);
  auto *entryBlock = func.addEntryBlock();
  auto entryBuilder = OpBuilder::atBlockBegin(entryBlock);
  bodyBuild(entryBuilder, loc, entryBlock->getArguments(), type.getResults());
  return func;
}

static void createUKernelGeneric(OpBuilder &moduleBuilder, Operation *op) {
  auto funcType = FunctionType::get(op->getContext(), op->getOperandTypes(),
                                    op->getResultTypes());
  llvm::StringRef opName = op->getName().getStringRef();
  auto func = createFuncOp(
      moduleBuilder, op->getLoc(), funcType, opName,
      [opName](OpBuilder &b, Location loc, ArrayRef<BlockArgument> operands,
               ArrayRef<Type> resultTypes) {
        Value firstOperand = operands[0];
        // TODO(ramiro050): check that operands are tensors
        RankedTensorType operandType =
            firstOperand.getType().cast<RankedTensorType>();
        Type elementType = operandType.getElementType();
        assert(operandType.getRank() == 1 &&
               "TODO(ramiro050): handle rank greater than 1");
        Value c0 = b.create<arith::ConstantIndexOp>(loc, 0);
        Value d0 = b.create<tensor::DimOp>(loc, firstOperand, c0);
        assert(resultTypes.size() == 1 &&
               "TODO(ramiro050): handle multiple returns");
        Value dest = b.create<tensor::EmptyOp>(loc, getAsOpFoldResult({d0}),
                                               elementType);

        auto dispatchRegion = b.create<IREE::Flow::DispatchRegionOp>(
            loc, resultTypes, /*result_dims=*/ValueRange{d0},
            /*workload=*/ValueRange{});
        Block &dispatchBody = dispatchRegion.getBody().emplaceBlock();
        {
          OpBuilder::InsertionGuard guard(b);
          b.setInsertionPointToStart(&dispatchBody);

          auto ukernel =
              b.create<IREE::Codegen::UKernelGenericOp>(
                   loc, resultTypes, (opName + "_workgroup").str(), operands,
                   dest, ValueRange{d0}, /*fn_def_attrs=*/nullptr,
                   /*strided_outer_dims=*/nullptr)
                  .getResults();

          b.create<Flow::ReturnOp>(loc, ukernel);
        }
        b.create<func::ReturnOp>(loc, dispatchRegion.getResults());
      });
  func.setPrivate();
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
    m.walk([&](Operation *op) {
      if (op->getDialect()->getNamespace() != "xnnpack") return;
      createUKernelGeneric(importBuilder, op);
      OpBuilder b(op);
      auto func =
          b.create<func::CallOp>(op->getLoc(), op->getName().getStringRef(),
                                 op->getResultTypes(), op->getOperands());
      op->replaceAllUsesWith(func.getResults());
      op->erase();
    });
  }
};

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> createLegalizeXnnpackPass() {
  return std::make_unique<LegalizeXnnpackPass>();
}

}  // namespace mlir::iree_compiler::IREE::Xnnpack
