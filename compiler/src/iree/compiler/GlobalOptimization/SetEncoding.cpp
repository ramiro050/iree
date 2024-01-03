// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===--------------- SetEncoding.cpp -------------------------------------===//
// Sets the encoding for compute operations to allow execution of the
// operations in tiled layouts.
//===---------------------------------------------------------------------===//

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree-dialects/Dialect/LinalgExt/Utils/Utils.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/GlobalOptimization/PassDetail.h"
#include "iree/compiler/GlobalOptimization/Passes.h"
#include "iree/compiler/GlobalOptimization/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Types.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/Support/Debug.h"

namespace mlir::iree_compiler::GlobalOptimization {

//===---------------------------------------------------------------------===//
// Utility functions
//===---------------------------------------------------------------------===//

/// Pads `value` enough for any actual tile sizes that could result from
/// materialization of `encodingAttr`.
static Value pad(OpBuilder &builder, Location loc, Value source,
                 IREE::LinalgExt::EncodingAttr encodingAttr) {
  RankedTensorType sourceType = source.getType().cast<RankedTensorType>();
  Type elemType = sourceType.getElementType();
  size_t rank = sourceType.getRank();
  RankedTensorType tensorTypeWithEncoding =
      RankedTensorType::get(sourceType.getShape(), elemType, encodingAttr);
  SmallVector<OpFoldResult> lowPad(rank, builder.getIndexAttr(0));
  SmallVector<Type> resultTypes(rank, builder.getIndexType());

  ValueRange encodingPaddingSizes =
      builder
          .create<IREE::LinalgExt::UpperBoundTileSizeOp>(
              loc, resultTypes, TypeAttr::get(tensorTypeWithEncoding))
          .getResults();
  SmallVector<OpFoldResult> highPad(rank);
  AffineExpr tileExpr, shapeExpr;
  bindSymbols(builder.getContext(), tileExpr, shapeExpr);
  AffineExpr highPadExpr = shapeExpr.ceilDiv(tileExpr) * tileExpr - shapeExpr;
  for (size_t i = 0; i < rank; ++i) {
    highPad[i] = affine::makeComposedFoldedAffineApply(
        builder, loc, highPadExpr,
        getAsOpFoldResult({encodingPaddingSizes[i],
                           builder.create<tensor::DimOp>(loc, source, i)}));
  }

  Value zero = builder.create<arith::ConstantOp>(loc, elemType,
                                                 builder.getZeroAttr(elemType));
  return builder.create<tensor::PadOp>(loc, /*resultType=*/nullptr, source,
                                       lowPad, highPad, zero);
}

Value setEncoding(OpBuilder &builder, Location loc, Value source,
                  IREE::LinalgExt::EncodingAttr encodingAttr) {
  auto sourceType = source.getType().cast<RankedTensorType>();
  auto resultType = RankedTensorType::get(
      sourceType.getShape(), sourceType.getElementType(), encodingAttr);
  return builder.create<IREE::LinalgExt::SetEncodingOp>(loc, resultType,
                                                        source);
};

enum class ContractionOpType {
  kInvalid,
  kMatmul,
  kBatchMatmul,
  kVecmat,
  kBatchVecmat,
  kMatvec,
  kBatchMatvec,
};

static ContractionOpType
getContractionOpType(linalg::ContractionOpInterface op) {
  if (op.isRowMajorMatmul() || op.isColumnMajorMatmul())
    return ContractionOpType::kMatmul;
  if (op.isRowMajorBatchMatmul())
    return ContractionOpType::kBatchMatmul;
  if (op.isVecmat())
    return ContractionOpType::kVecmat;
  if (op.isBatchVecmat())
    return ContractionOpType::kBatchVecmat;
  if (op.isMatvec())
    return ContractionOpType::kMatvec;
  if (op.isBatchMatvec())
    return ContractionOpType::kBatchMatvec;
  return ContractionOpType::kInvalid;
}

struct MatmulNarrowSizes {
  std::optional<int64_t> M, N;
};

// Returns the minimum of static sizes of the M/N-dimensions in the types of the
// Ouput.
static MatmulNarrowSizes getMatmulNarrowSizes(ShapedType outType,
                                              ContractionOpType opType) {
  int64_t M, N;
  int64_t rank = outType.getRank();
  switch (opType) {
  case ContractionOpType::kMatmul:
  case ContractionOpType::kBatchMatmul: {
    M = outType.getDimSize(rank - 2);
    N = outType.getDimSize(rank - 1);
    break;
  }
  case ContractionOpType::kVecmat:
  case ContractionOpType::kBatchVecmat: {
    M = 1;
    N = outType.getDimSize(outType.getRank() - 1);
    break;
  }
  case ContractionOpType::kMatvec:
  case ContractionOpType::kBatchMatvec: {
    M = outType.getDimSize(outType.getRank() - 1);
    N = 1;
    break;
  }
  case ContractionOpType::kInvalid:
    return MatmulNarrowSizes();
  }

  MatmulNarrowSizes narrow;
  // Threshold below which a M/N size is considered "narrow", making it
  // eligible for a narrow tile size during materialization. This value should
  // be at least as large as the actual M/N tile sizes that we choose on any
  // target in CPUMaterializeEncodingPass. If it is smaller, we will miss
  // opportunities to select optimized narrow tiles for narrow matmuls.
  // If it is larger, everything will work fine, but the IR will be a bit more
  // verbose as more narrow_matmul_{M,N} optional parameters will be specified.
  const int64_t kNarrowThreshold = 16;
  if (!ShapedType::isDynamic(M) && M < kNarrowThreshold) {
    narrow.M = M;
  }
  if (!ShapedType::isDynamic(N) && N < kNarrowThreshold) {
    narrow.N = N;
  }
  return narrow;
}

static IREE::LinalgExt::EncodingAttr
makeEncoding(OpBuilder &builder, IREE::LinalgExt::EncodingUser user,
             IREE::LinalgExt::EncodingRole role, TypeRange operandTypes,
             Type originalType, MatmulNarrowSizes narrow) {
  auto *context = builder.getContext();
  auto userAttr = IREE::LinalgExt::EncodingUserAttr::get(context, user);
  auto roleAttr = IREE::LinalgExt::EncodingRoleAttr::get(context, role);
  SmallVector<Attribute> elemTypeAttrs =
      llvm::map_to_vector(operandTypes, [](auto t) {
        return TypeAttr::get(t.template cast<ShapedType>().getElementType())
            .template cast<Attribute>();
      });
  auto operandElemTypesAttr = ArrayAttr::get(context, elemTypeAttrs);
  auto originalTypeAttr =
      originalType ? TypeAttr::get(originalType) : TypeAttr{};
  auto getAttr = [&](std::optional<int64_t> x) {
    return x ? builder.getIndexAttr(*x) : IntegerAttr();
  };
  return IREE::LinalgExt::EncodingAttr::get(
      context, userAttr, roleAttr, operandElemTypesAttr, originalTypeAttr,
      getAttr(narrow.M), getAttr(narrow.N));
}

// Creates a linalg::GenericOp that performs an element-wise cast of the same
// type as performed in `castOp`, and returns the result enceoded with
// `encodingAttr`. The element type of `encoded` is expected to be the same as
// the element type of the input to `castOp`, which can be a CastOpInterface op
// on a tensor or single element.
static Value castEncodedResult(OpBuilder &builder, Location loc, Value encoded,
                               CastOpInterface castOp,
                               IREE::LinalgExt::EncodingAttr encodingAttr) {
  auto genericOp = castOp->getParentOfType<linalg::GenericOp>();
  NamedAttrList castAttrs = genericOp
                                ? linalg::getPrunedAttributeList(genericOp)
                                : castOp->getAttrs();
  return createGenericElementwiseCastOp(builder, loc, encoded, castOp,
                                        castAttrs, encodingAttr);
}

static Value
padAndSetEncoding(OpBuilder &builder, Location loc, Value source,
                  IREE::LinalgExt::EncodingUser user,
                  IREE::LinalgExt::EncodingRole role, TypeRange operandTypes,
                  MatmulNarrowSizes narrow,
                  std::optional<CastOpInterface> castOp = std::nullopt) {
  Value padSource = castOp ? source.getDefiningOp()->getOperand(0) : source;
  // No need to specify original_type in the encoding poadded to pad(), because
  // the operand there is the `source` tensor, so it will default to reading its
  // original shape.
  auto encodingForPad = makeEncoding(builder, user, role, operandTypes,
                                     /*originalType=*/Type{}, narrow);
  Value padded = pad(builder, loc, padSource, encodingForPad);
  // For setEncoding() below, we potentially need to specify an encoding with an
  // explicit original_type, because the operand there is the padded tensor
  // returned by pad() above, but we want setEncoding to be aware of the
  // original source tensor shape, not the padded tensor shape. To limit IR
  // verbosity, we only specify the original original_type when it differs from
  // the tensor type that the encoding is applied to.
  auto encodingForSetEncoding = encodingForPad;
  if (padded.getType() != padSource.getType()) {
    encodingForSetEncoding = makeEncoding(builder, user, role, operandTypes,
                                          padSource.getType(), narrow);
  }
  Value encoded = setEncoding(builder, loc, padded, encodingForSetEncoding);
  if (castOp) {
    encoded = castEncodedResult(builder, loc, encoded, castOp.value(),
                                encodingForSetEncoding);
  }
  return encoded;
}

static Value unsetEncodingAndExtractSlice(OpBuilder &builder, Location loc,
                                          Value source,
                                          SmallVector<OpFoldResult> sizes) {
  auto sourceType = source.getType().cast<RankedTensorType>();
  auto unsetEncodingReturnType =
      RankedTensorType::get(sourceType.getShape(), sourceType.getElementType());
  auto unsetEncoding = builder
                           .create<IREE::LinalgExt::UnsetEncodingOp>(
                               loc, unsetEncodingReturnType, source)
                           .getResult();
  auto rank = sourceType.getRank();
  SmallVector<OpFoldResult> offsets(rank, builder.getIndexAttr(0));
  SmallVector<OpFoldResult> strides(rank, builder.getIndexAttr(1));
  return builder.create<tensor::ExtractSliceOp>(loc, unsetEncoding, offsets,
                                                sizes, strides);
}

namespace {

struct setContractionOpEncoding
    : public OpInterfaceRewritePattern<linalg::ContractionOpInterface> {
  using OpInterfaceRewritePattern<
      linalg::ContractionOpInterface>::OpInterfaceRewritePattern;
  LogicalResult matchAndRewrite(linalg::ContractionOpInterface op,
                                PatternRewriter &rewriter) const override {
    auto linalgOp = dyn_cast<linalg::LinalgOp>(op.getOperation());
    if (!linalgOp.hasTensorSemantics()) {
      return failure();
    }
    if (getCompilationInfo(linalgOp)) {
      return rewriter.notifyMatchFailure(
          linalgOp, "the op has preset compilation strategy, skip SetEncoding");
    }

    auto inputs = linalgOp.getDpsInputs();
    auto outputs = linalgOp.getDpsInits();

    auto hasEncoding = [](Value operand) -> bool {
      auto type = llvm::dyn_cast<RankedTensorType>(operand.getType());
      return type && type.getEncoding();
    };
    if (llvm::any_of(inputs, hasEncoding) ||
        llvm::any_of(outputs, hasEncoding)) {
      return failure();
    }
    Value origLhs = inputs[0];
    Value origRhs = inputs[1];
    Value origOut = outputs[0];

    auto getElemType = [](Value v) -> Type {
      if (auto tensorType = llvm::dyn_cast<RankedTensorType>(v.getType())) {
        return tensorType.getElementType();
      }
      return {};
    };
    std::optional<CastOpInterface> maybeLhsCastOp =
        getDefiningNonI1ExtendingCastOp(origLhs);
    std::optional<CastOpInterface> maybeRhsCastOp =
        getDefiningNonI1ExtendingCastOp(origRhs);
    Type lhsElemType = maybeLhsCastOp ? getCastElemType(origLhs).value()
                                      : getElemType(origLhs);
    Type rhsElemType = maybeRhsCastOp ? getCastElemType(origRhs).value()
                                      : getElemType(origRhs);
    Type outElemType = getElemType(origOut);

    if (!lhsElemType || !rhsElemType || !outElemType) {
      return failure();
    }

    ContractionOpType opType = getContractionOpType(op);
    IREE::LinalgExt::EncodingUser user;
    switch (opType) {
    case ContractionOpType::kMatmul:
    case ContractionOpType::kVecmat:
    case ContractionOpType::kMatvec:
      user = IREE::LinalgExt::EncodingUser::MATMUL;
      break;
    case ContractionOpType::kBatchMatmul:
    case ContractionOpType::kBatchVecmat:
    case ContractionOpType::kBatchMatvec:
      user = IREE::LinalgExt::EncodingUser::BATCH_MATMUL;
      break;
    case ContractionOpType::kInvalid:
      return rewriter.notifyMatchFailure(op, "unsupported contraction op");
    }

    MatmulNarrowSizes narrowSizes =
        getMatmulNarrowSizes(origOut.getType().cast<ShapedType>(), opType);

    Location loc = linalgOp.getLoc();
    SmallVector<Type> operandTypes(linalgOp->getOperandTypes());
    operandTypes[0] =
        cast<RankedTensorType>(operandTypes[0]).clone(lhsElemType);
    operandTypes[1] =
        cast<RankedTensorType>(operandTypes[1]).clone(rhsElemType);
    Value encodedLhs = padAndSetEncoding(
        rewriter, loc, origLhs, user, IREE::LinalgExt::EncodingRole::LHS,
        operandTypes, narrowSizes, maybeLhsCastOp);
    Value encodedRhs = padAndSetEncoding(
        rewriter, loc, origRhs, user, IREE::LinalgExt::EncodingRole::RHS,
        operandTypes, narrowSizes, maybeRhsCastOp);
    Value encodedOut = padAndSetEncoding(rewriter, loc, origOut, user,
                                         IREE::LinalgExt::EncodingRole::RESULT,
                                         operandTypes, narrowSizes);
    Value opTiled;
    opTiled = clone(rewriter, linalgOp, encodedOut.getType(),
                    ValueRange{encodedLhs, encodedRhs, encodedOut})
                  ->getResult(0);

    // Sizes are computed by original output size.
    FailureOr<SmallVector<OpFoldResult>> origOutSizes =
        IREE::LinalgExt::getDims(rewriter, loc, origOut);
    if (failed(origOutSizes)) {
      return rewriter.notifyMatchFailure(linalgOp,
                                         "failed to get shape of result");
    }

    Value result = unsetEncodingAndExtractSlice(rewriter, loc, opTiled,
                                                origOutSizes.value());

    rewriter.replaceOp(linalgOp, result);
    return success();
  }
};

/// Pattern to fold a `linalg.fill` -> `iree_linalg_ext.set_encoding`
/// operation into a `linalg.fill` of the encoded type.
struct FoldFillWithSetEncoding
    : public OpRewritePattern<IREE::LinalgExt::SetEncodingOp> {
  using OpRewritePattern<IREE::LinalgExt::SetEncodingOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(IREE::LinalgExt::SetEncodingOp encodingOp,
                                PatternRewriter &rewriter) const override {
    auto fillOp = encodingOp.getSource().getDefiningOp<linalg::FillOp>();
    if (!fillOp)
      return failure();

    // Create a new fill op, with outs being defined by a new `tensor.empty` op.
    RankedTensorType encodingType = encodingOp.getResultType();
    Location loc = fillOp.getLoc();
    SmallVector<OpFoldResult> dimValues =
        tensor::getMixedSizes(rewriter, loc, fillOp.getOutputs()[0]);
    auto newEmptyOp = rewriter.create<tensor::EmptyOp>(
        loc, dimValues, encodingType.getElementType(),
        encodingType.getEncoding());
    rewriter.replaceOpWithNewOp<linalg::FillOp>(encodingOp, fillOp.getInputs(),
                                                ValueRange{newEmptyOp});
    return success();
  }
};

struct SetEncodingPass : public SetEncodingBase<SetEncodingPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::LinalgExt::IREELinalgExtDialect>();
  }

  void runOnOperation() override;
};
} // namespace

void SetEncodingPass::runOnOperation() {
  MLIRContext *context = &getContext();
  {
    RewritePatternSet patterns(context);
    patterns.insert<setContractionOpEncoding>(context);
    linalg::FillOp::getCanonicalizationPatterns(patterns, context);
    patterns.insert<FoldFillWithSetEncoding>(context);
    memref::populateResolveRankedShapedTypeResultDimsPatterns(patterns);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
}

std::unique_ptr<Pass> createSetEncodingPass() {
  return std::make_unique<SetEncodingPass>();
}

} // namespace mlir::iree_compiler::GlobalOptimization
