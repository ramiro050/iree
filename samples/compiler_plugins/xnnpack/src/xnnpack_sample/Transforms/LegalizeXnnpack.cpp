// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

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
    OpBuilder &b, Location loc, FunctionType type, std::string name,
    function_ref<void(OpBuilder &, Location, ArrayRef<BlockArgument>,
                      ArrayRef<Type>)>
        bodyBuild) {
  auto func = b.create<func::FuncOp>(loc, name, type);
  auto *entryBlock = func.addEntryBlock();
  auto entryBuilder = OpBuilder::atBlockBegin(entryBlock);
  bodyBuild(entryBuilder, loc, entryBlock->getArguments(), type.getResults());
  return func;
}

static LogicalResult defineUKernelCall(ModuleOp m, Operation *op) {
  auto importBuilder = OpBuilder::atBlockBegin(m.getBody());
  auto funcType = FunctionType::get(op->getContext(), op->getOperandTypes(),
                                    op->getResultTypes());
  auto func = createFuncOp(
      importBuilder, op->getLoc(), funcType, "xnnpack.mul2",
      [](OpBuilder &b, Location loc, ArrayRef<BlockArgument> operands,
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

          b.create<IREE::Codegen::UKernelGenericOp>(
              loc, resultTypes, "xnnpack_mul2_workgroup", operands, dest,
              ValueRange{d0}, /*fn_def_attrs=*/nullptr,
              /*strided_outer_dims=*/nullptr);

          b.create<Flow::ReturnOp>(loc, dest);
        }
        b.create<func::ReturnOp>(loc, dest);
      });
  func.setPrivate();
  return success();
}

class LegalizeXnnpackPass
    : public ::impl::LegalizeXnnpackBase<LegalizeXnnpackPass> {
 public:
  void runOnOperation() override {
    auto *context = &getContext();
    // TODO: This is all just a placeholder. To make it real, we should be
    // checking if the import already exists and likely doing some more fancy
    // lowering.
    // Add imports.
    auto m = getOperation();
    auto importBuilder = OpBuilder::atBlockBegin(m.getBody());
    importBuilder
        .create<func::FuncOp>(m.getLoc(), "xnnpack.print",
                              FunctionType::get(context, {}, {}))
        .setPrivate();

    // Legalize operations.
    m.walk([&](Operation *op) {
      if (auto printOp = dyn_cast<IREE::Xnnpack::PrintOp>(op)) {
        OpBuilder b(op);
        b.create<func::CallOp>(printOp.getLoc(), "xnnpack.print", TypeRange{});
        printOp.erase();
      } else if (auto mul2Op = dyn_cast<IREE::Xnnpack::Mul2Op>(op)) {
        // TODO(ramiro050): handle this properly
        if (failed(defineUKernelCall(m, op))) return;
        OpBuilder b(op);
        Value func =
            b.create<func::CallOp>(mul2Op.getLoc(), "xnnpack.mul2",
                                   mul2Op.getType(), mul2Op.getOperands())
                .getResult(0);
        mul2Op.replaceAllUsesWith(func);
        mul2Op.erase();
      }
    });
  }
};

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> createLegalizeXnnpackPass() {
  return std::make_unique<LegalizeXnnpackPass>();
}

}  // namespace mlir::iree_compiler::IREE::Xnnpack
