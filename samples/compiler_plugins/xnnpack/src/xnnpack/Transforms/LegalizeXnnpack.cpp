// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/UKernelOps.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "xnnpack/IR/XnnpackOps.h"
#include "xnnpack/Transforms/Passes.h"

namespace mlir::iree_compiler::IREE::Xnnpack {
#define GEN_PASS_DEF_LEGALIZEXNNPACK
#include "xnnpack/Transforms/Passes.h.inc"

namespace {
static FailureOr<func::FuncOp> createFuncOp(
    RewriterBase &rewriter, Location loc, FunctionType type,
    llvm::StringRef name,
    function_ref<LogicalResult(RewriterBase &, Location,
                               ArrayRef<BlockArgument>, ArrayRef<Type>)>
        bodyBuild) {
  auto func = rewriter.create<func::FuncOp>(loc, name, type);
  auto *entryBlock = func.addEntryBlock();
  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(entryBlock);
    if (failed(bodyBuild(rewriter, loc, entryBlock->getArguments(),
                         type.getResults()))) {
      // `bodyBuild` can leave the `FuncOp` in an invalid state, so erase it
      // before returning.
      rewriter.eraseOp(func);
      return failure();
    }
  }
  return func;
}

// For every operand in `funcOpOperands`, returns a vector with the operands
// sizes, folded if sizes are static.
//
// TODO(ramiro050): this can be done a bit smarter without so much dynamic
// casting. One option is to have a separate pass that attaches the result
// dimension information as an attribute to each op.
// TODO(ramiro050): improve name of `funcOpOperands`
static FailureOr<SmallVector<SmallVector<Value>>> getInputOutputDims(
    Operation *op, ArrayRef<BlockArgument> funcOpOperands, OpBuilder &b) {
  auto getDim = [&](Value tensor, int64_t dim) -> Value {
    Value ci = b.create<arith::ConstantIndexOp>(op->getLoc(), dim);
    Value dimValue = b.createOrFold<tensor::DimOp>(op->getLoc(), tensor, ci);
    return dimValue;
  };
  auto getDims = [&](Value tensor) -> SmallVector<Value> {
    auto tensorType = tensor.getType().cast<RankedTensorType>();
    int64_t rank = tensorType.getRank();

    SmallVector<Value> dims;
    for (auto i : llvm::seq(rank)) dims.push_back(getDim(tensor, i));
    return dims;
  };

  SmallVector<SmallVector<Value>> dims;
  for (Value operand : funcOpOperands) {
    dims.push_back(getDims(operand));
  }

  if (auto matmul = dyn_cast<BatchMatrixMultiplyOp>(op)) {
    int64_t rankA = matmul.getA().getType().cast<RankedTensorType>().getRank();
    int64_t rankB = matmul.getB().getType().cast<RankedTensorType>().getRank();
    if (rankA != 3 || rankB != 3) {
      return op->emitError("unimplemented: matmul arguments with rank != 3");
    }
    dims.push_back({dims[0][0], dims[0][1], dims[1][2]});
  } else if (auto mul2 = dyn_cast<Multiply2Op>(op)) {
    // TODO(ramiro050): I think this is supposed to do broadcasting.
    int64_t rankA = mul2.getA().getType().cast<RankedTensorType>().getRank();
    int64_t rankB = mul2.getB().getType().cast<RankedTensorType>().getRank();
    if (rankA != rankB) {
      return op->emitError("unimplemented: broadcasting");
    }
    dims.push_back({dims[0][0]});
  } else if (auto fullyConnected =
                 dyn_cast<FullyConnectedNcQd8F32Qc4w_Rank3InputOp>(op)) {
    auto inputType =
        fullyConnected.getInput().getType().cast<RankedTensorType>();
    ArrayRef<int64_t> batchDims = inputType.getShape().drop_back(2);
    auto kernelType =
        fullyConnected.getKernel().getType().cast<RankedTensorType>();
    if (inputType.getRank() > 2) {
      for (int64_t batchDim : batchDims) {
        if (batchDim != 1) {
          return op->emitError(
              "unimplemented: input with batch dimensions of size != 1");
        }
      }
    }
    if (kernelType.getRank() != 2) {
      return op->emitError("unimplemented: kernel of rank != 2");
    }

    ArrayRef<Value> inputDims(dims[0]);
    ArrayRef<Value> kernelDims(dims[1]);
    // Fully connected performs a reduction along the outer dimension of the
    // `kernel` tensor when no transpose is needed for the `kernel` tensor, and
    // a reduction along the inner dimension otherwise.
    SmallVector<Value> outputDims(inputDims.drop_back(1));
    outputDims.push_back(fullyConnected.getTransposeRhs() ? kernelDims[1]
                                                          : kernelDims[0]);
    dims.push_back(outputDims);
  } else if (auto fullyConnected =
                 dyn_cast<FullyConnectedNcQd8F32Qc4w_Rank2InputOp>(op)) {
    auto inputType =
        fullyConnected.getInput().getType().cast<RankedTensorType>();
    auto kernelType =
        fullyConnected.getKernel().getType().cast<RankedTensorType>();
    if (inputType.getRank() != 2) return op->emitError("input must be rank 2");
    if (kernelType.getRank() != 2)
      return op->emitError("unimplemented: kernel of rank != 2");

    ArrayRef<Value> inputDims(dims[0]);
    ArrayRef<Value> kernelDims(dims[1]);
    // Fully connected performs a reduction along the outer dimension of the
    // `kernel` tensor when no transpose is needed for the `kernel` tensor, and
    // a reduction along the inner dimension otherwise.
    SmallVector<Value> outputDims{inputDims[0]};
    outputDims.push_back(fullyConnected.getTransposeRhs() ? kernelDims[1]
                                                          : kernelDims[0]);
    dims.push_back(outputDims);
  } else {
    llvm_unreachable("not an xnnpack op!");
  }
  return dims;
}

static FailureOr<func::FuncOp> createUKernelGeneric(
    RewriterBase &moduleRewriter, Operation *op, int uniqueId) {
  auto funcType = FunctionType::get(op->getContext(), op->getOperandTypes(),
                                    op->getResultTypes());
  llvm::StringRef opName = op->getName().getStringRef();
  auto func = createFuncOp(
      moduleRewriter, op->getLoc(), funcType,
      (opName + "_" + std::to_string(uniqueId)).str(),
      [opName, op](RewriterBase &rewriter, Location loc,
                   ArrayRef<BlockArgument> operands,
                   ArrayRef<Type> resultTypes) -> LogicalResult {
        if (llvm::any_of(operands, [](BlockArgument operand) {
              return !operand.getType().dyn_cast<RankedTensorType>();
            })) {
          return op->emitError("unimplemented: non-tensor argument");
        }

        if (resultTypes.size() != 1) {
          return op->emitError("unimplemented: multiple returns");
        }
        Type resultElementType =
            resultTypes[0].cast<RankedTensorType>().getElementType();
        FailureOr<SmallVector<SmallVector<Value>>> maybeDims(
            getInputOutputDims(op, operands, rewriter));
        if (failed(maybeDims)) return failure();
        ArrayRef<Value> resultDims(maybeDims.value().back());
        SmallVector<OpFoldResult> resultDimsFolded(
            getAsOpFoldResult(resultDims));
        auto [_, dynResultDims] =
            decomposeMixedValues(rewriter, resultDimsFolded);

        Value dest = rewriter.create<tensor::EmptyOp>(loc, resultDimsFolded,
                                                      resultElementType);

        auto dispatchRegion = rewriter.create<IREE::Flow::DispatchRegionOp>(
            loc, resultTypes, /*result_dims=*/dynResultDims,
            /*workload=*/ValueRange{});
        Block &dispatchBody = dispatchRegion.getBody().emplaceBlock();
        {
          OpBuilder::InsertionGuard guard(rewriter);
          rewriter.setInsertionPointToStart(&dispatchBody);

          SmallVector<Value> otherOperands;
          for (auto dims : maybeDims.value()) otherOperands.append(dims);
          for (auto attr : op->getAttrs()) {
            if (auto boolAttr = dyn_cast<BoolAttr>(attr.getValue())) {
              Value boolVal = rewriter.create<arith::ConstantIntOp>(
                  loc, boolAttr.getValue(), 8);
              otherOperands.push_back(boolVal);
            } else {
              return op->emitError()
                     << "unimplemented: handling of attribute with value '"
                     << attr.getValue() << "'";
            }
          }

          auto ukernel =
              rewriter
                  .create<IREE::Codegen::UKernelGenericOp>(
                      loc, resultTypes, (opName + "_workgroup").str(), operands,
                      dest, otherOperands,
                      /*fn_def_attrs=*/nullptr,
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
    : public impl::LegalizeXnnpackBase<LegalizeXnnpackPass> {
 public:
  LegalizeXnnpackPass() = default;
  LegalizeXnnpackPass(const LegalizeXnnpackPass &) {}
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<tensor::TensorDialect, IREE::Flow::FlowDialect,
                    IREE::Codegen::IREECodegenDialect>();
  }
  void runOnOperation() override {
    auto m = getOperation();
    auto importBuilder = OpBuilder::atBlockBegin(m.getBody());
    IRRewriter rewriter(importBuilder);
    // IREE expects that the name of each function declaration is unique. To
    // ensure this, the value of `unique_id` is appended to the name of the
    // function.
    // Code bloat is not a big concern because IREE makes an attempt at
    // deduplicates functions.
    // TODO: Use a map to keep track of the available functions and avoid
    // creating a new function when the same one already exists.
    int uniqueId = 0;
    auto result = m.walk([&](Operation *op) -> WalkResult {
      if (op->getDialect()->getNamespace() != "xnnpack")
        return WalkResult::advance();

      FailureOr<func::FuncOp> ukernelGeneric =
          createUKernelGeneric(rewriter, op, uniqueId++);
      if (failed(ukernelGeneric)) return WalkResult::interrupt();
      ukernelGeneric->setPrivate();

      {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPoint(op);
        rewriter.replaceOpWithNewOp<func::CallOp>(op, ukernelGeneric->getName(),
                                                  op->getResultTypes(),
                                                  op->getOperands());
      }
      return WalkResult::advance();
    });
    if (result.wasInterrupted()) signalPassFailure();
  }
};

}  // namespace

}  // namespace mlir::iree_compiler::IREE::Xnnpack
