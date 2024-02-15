// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <numeric>

#include "iree-dialects/Dialect/VectorExt/IR/VectorExtDialect.h"
#include "iree-dialects/Dialect/VectorExt/IR/VectorExtOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;

namespace mlir::iree_compiler::IREE::VectorExt {

bool PerDimLayoutAttr::contains(const LayoutDimension &dim) {
  for (LayoutDimensionAttr label : getLabels()) {
    if (label.getValue() == dim)
      return true;
  }
  return false;
}

std::optional<int64_t> PerDimLayoutAttr::getShape(const LayoutDimension &dim) {
  for (auto value : llvm::zip(getLabels(), getShapes())) {
    if (dim == std::get<0>(value).getValue())
      return std::get<1>(value);
  }
  return std::nullopt;
}

std::optional<int64_t> LayoutAttr::getShape(const LayoutDimension &dim) const {
  for (PerDimLayoutAttr layout : getLayouts()) {
    std::optional<int64_t> maybeShape = layout.getShape(dim);
    if (maybeShape)
      return maybeShape.value();
  }
  return std::nullopt;
}

// Get the SIMT Vector shape in the order specified by dims. If no dims are
// specified, then return an empty vector.
bool LayoutAttr::isValidLayout(ArrayRef<int64_t> shape) const {
  for (auto perDimLayout : llvm::enumerate(getLayouts())) {
    ArrayRef<int64_t> layoutShape = perDimLayout.value().getShapes();
    int64_t computedShape =
        std::reduce(layoutShape.begin(), layoutShape.end(),
                    static_cast<int64_t>(1), std::multiplies<int64_t>());
    int64_t expectedShape = shape[perDimLayout.index()];
    if (computedShape != expectedShape) {
      return false;
    }
  }
  return true;
}

// Project out the layout for the specified dimensions
// resulting in the layout for a lower dimensional vector.
VectorLayoutInterface LayoutAttr::project(ArrayRef<bool> projectedDims) const {
  assert(projectedDims.size() == getLayouts().size() &&
         "projectedDims size must match layout size");

  ArrayRef<PerDimLayoutAttr> layouts = getLayouts();
  assert(projectedDims.size() == layouts.size());
  SmallVector<PerDimLayoutAttr> newLayouts;
  for (auto pair : llvm::zip(projectedDims, layouts)) {
    if (!std::get<0>(pair))
      newLayouts.push_back(std::get<1>(pair));
  }
  return LayoutAttr::get(getContext(), newLayouts);
}

// Permute the layout according to the provided permutation
// vector. The dimensionality of the layout remains the same.
VectorLayoutInterface LayoutAttr::permute(ArrayRef<int64_t> permutation) const {
  assert(permutation.size() == getLayouts().size() &&
         "permutation size must match layout size");

  ArrayRef<PerDimLayoutAttr> layouts = getLayouts();
  assert(permutation.size() == layouts.size());
  SmallVector<PerDimLayoutAttr> newLayouts;
  for (unsigned index : permutation) {
    assert(index >= 0 && index < layouts.size());
    newLayouts.push_back(layouts[index]);
  }
  return LayoutAttr::get(getContext(), newLayouts);
}

// This function returns the distributed shape of the SIMT
// vector and evaluates it in the following order:
// BATCHX, BATCHY, VECTORY, VECTORX
// The vector dimensions are combined into a single SIMT
// vector dimension.
SmallVector<int64_t> LayoutAttr::getDistributedShape() const {
  SmallVector<LayoutDimension> labels{
      LayoutDimension::BATCHX, LayoutDimension::BATCHY,
      LayoutDimension::VECTORY, LayoutDimension::VECTORX};
  SmallVector<int64_t> simtVectorShape;
  std::optional<int64_t> vectorShape;
  for (LayoutDimension dim : labels) {
    ArrayRef<PerDimLayoutAttr> layouts = getLayouts();
    for (PerDimLayoutAttr layout : layouts) {
      if (!layout.contains(dim))
        continue;
      int64_t shape = layout.getShape(dim).value();
      if (isVectorDimension(dim)) {
        vectorShape = shape * vectorShape.value_or(1);
        continue;
      }
      simtVectorShape.push_back(shape);
    }
  }
  if (vectorShape)
    simtVectorShape.push_back(vectorShape.value());
  return simtVectorShape;
}

PerDimLayoutAttr LayoutAttr::getDimLayout(int64_t dim) const {
  assert(dim >= 0 && dim < getLayouts().size());
  return getLayouts()[dim];
}

std::optional<int64_t> LayoutAttr::getBatchDim(int64_t dim) {
  assert(dim < getLayouts().size());
  PerDimLayoutAttr layout = getDimLayout(dim);
  for (auto [name, shape] :
       llvm::zip_equal(layout.getLabels(), layout.getShapes())) {
    if (isBatchDimension(name.getValue()))
      return shape;
  }
  return std::nullopt;
}

std::optional<int64_t> LayoutAttr::getLaneDim(int64_t dim) {
  assert(dim < getLayouts().size());
  PerDimLayoutAttr layout = getDimLayout(dim);
  for (auto [name, shape] :
       llvm::zip_equal(layout.getLabels(), layout.getShapes())) {
    if (isLaneDimension(name.getValue()))
      return shape;
  }
  return std::nullopt;
}

std::optional<LayoutDimension> LayoutAttr::getLane(int64_t dim) {
  assert(dim < getLayouts().size());
  PerDimLayoutAttr layout = getDimLayout(dim);
  for (auto [name, shape] :
       llvm::zip_equal(layout.getLabels(), layout.getShapes())) {
    if (isLaneDimension(name.getValue()))
      return name.getValue();
  }
  return std::nullopt;
}

std::tuple<int64_t, int64_t, int64_t> LayoutAttr::getLaneGrid() {
  int64_t laneX = 1;
  int64_t laneY = 1;
  int64_t laneZ = 1;
  for (PerDimLayoutAttr dimLayout : getLayouts()) {
    // Note that valid layouts only include at most one instance of each
    // dimension type, so this is simply doing assignment on the first instance
    // of each lane index, not an accumulative product.
    auto maybeXShape = dimLayout.getShape(LayoutDimension::LANEX);
    laneX *= maybeXShape.value_or(1);
    auto maybeYShape = dimLayout.getShape(LayoutDimension::LANEY);
    laneY *= maybeYShape.value_or(1);
    auto maybeZShape = dimLayout.getShape(LayoutDimension::LANEZ);
    laneZ *= maybeZShape.value_or(1);
  }
  return std::make_tuple(laneX, laneY, laneZ);
}

uint64_t LayoutAttr::getShuffleOffset(int64_t reductionDim) {
  uint64_t offset = 0;
  std::optional<LayoutDimension> laneDim = getLane(reductionDim);
  if (!laneDim)
    return offset;
  switch (laneDim.value()) {
  case LayoutDimension::LANEX:
    offset = 1;
    break;
  case LayoutDimension::LANEY:
    offset = getShape(LayoutDimension::LANEX).value_or(0);
    break;
  case LayoutDimension::LANEZ:
    offset = getShape(LayoutDimension::LANEX).value_or(0) *
             getShape(LayoutDimension::LANEY).value_or(0);
    break;
  default:
    assert("Invalid dimension! Expected lane dimension.");
    break;
  }
  return offset;
}

bool LayoutAttr::hasLaneConflictWith(const LayoutAttr &other) {
  SmallVector<LayoutDimension> laneDims{
      LayoutDimension::LANEX, LayoutDimension::LANEY, LayoutDimension::LANEZ};
  for (LayoutDimension dim : laneDims) {
    std::optional<int64_t> shape = getShape(dim);
    std::optional<int64_t> otherShape = other.getShape(dim);
    if ((shape && !otherShape) || (!shape && otherShape))
      return true;
    if (shape && otherShape) {
      if (shape.value() != otherShape.value())
        return true;
    }
  }
  return false;
}

VectorLayoutInterface
NestedLayoutAttr::project(ArrayRef<bool> projectedDims) const {
  llvm_unreachable("Not yet implemented");
}

VectorLayoutInterface
NestedLayoutAttr::permute(ArrayRef<int64_t> permutation) const {
  llvm_unreachable("Not yet implemented");
}

/// We distribute to:
/// <BATCH x OUTER x ELEMENT>
SmallVector<int64_t> NestedLayoutAttr::getDistributedShape() const {
  SmallVector<int64_t> shape;
  shape.append(applyPermutation(getBatchesPerSubgroup(), getBatchOrder()));
  shape.append(applyPermutation(getOutersPerBatch(), getOuterOrder()));
  shape.append(applyPermutation(getElementsPerThread(), getElementOrder()));
  return shape;
}

bool NestedLayoutAttr::isValidLayout(ArrayRef<int64_t> shape) const {
  // Multiply all shapes in the layout.
  for (int i = 0, e = shape.size(); i < e; ++i) {
    int64_t expectedShape = getSubgroupsPerWorkgroup()[i] *
                            getBatchesPerSubgroup()[i] *
                            getOutersPerBatch()[i] * getThreadsPerOuter()[i] *
                            getElementsPerThread()[i];
    if (expectedShape != shape[i]) {
      return false;
    }
  }
  return true;
}

// TODO: These things should ideally go into the parser when we have a custom
// parser.
LogicalResult NestedLayoutAttr::verify(
    llvm::function_ref<InFlightDiagnostic()> emitError,
    ArrayRef<int64_t> subgroupsPerWorkgroup, ArrayRef<int64_t> subgroupOrder,
    ArrayRef<int64_t> batchesPerSubgroup, ArrayRef<int64_t> batchOrder,
    ArrayRef<int64_t> outersPerBatch, ArrayRef<int64_t> outerOrder,
    ArrayRef<int64_t> threadsPerOuter, ArrayRef<int64_t> threadOrder,
    ArrayRef<int64_t> elementsPerThread, ArrayRef<int64_t> elementOrder) {

  size_t rank = subgroupsPerWorkgroup.size();
  auto checkTile = [&](ArrayRef<int64_t> tileShape, ArrayRef<int64_t> order) {
    if (tileShape.size() != rank || order.size() != rank) {
      emitError() << "all tiles must have the same rank as the layout";
      return failure();
    }
    if (!mlir::isPermutationVector(order)) {
      emitError() << "all orderings must be permutation vectors";
      return failure();
    }
    return success();
  };

  if (failed(checkTile(subgroupsPerWorkgroup, subgroupOrder)) ||
      failed(checkTile(batchesPerSubgroup, batchOrder)) ||
      failed(checkTile(outersPerBatch, outerOrder)) ||
      failed(checkTile(threadsPerOuter, threadOrder)) ||
      failed(checkTile(elementsPerThread, elementOrder))) {
    return failure();
  }

  return success();
}

/// Given a single flat thread ID, compute the indices of the distributed
/// dimensions (subgroup and thread ids).
ValueRange NestedLayoutAttr::computeThreadIds(Value threadId,
                                              RewriterBase &rewriter) const {
  SmallVector<OpFoldResult> basis;
  for (auto warpTy : getSubgroupOrder()) {
    basis.push_back(rewriter.getIndexAttr(getSubgroupsPerWorkgroup()[warpTy]));
  }
  for (auto threadTy : getThreadOrder()) {
    basis.push_back(rewriter.getIndexAttr(getThreadsPerOuter()[threadTy]));
  }

  auto delinearized = rewriter.create<mlir::affine::AffineDelinearizeIndexOp>(
      threadId.getLoc(), threadId, basis);
  return delinearized->getResults();
}

} // namespace mlir::iree_compiler::IREE::VectorExt

using namespace mlir::iree_compiler::IREE::VectorExt;

#include "iree-dialects/Dialect/VectorExt/IR/VectorExtEnums.cpp.inc" // IWYU pragma: keep

#define GET_ATTRDEF_CLASSES
#include "iree-dialects/Dialect/VectorExt/IR/VectorExtAttrs.cpp.inc" // IWYU pragma: keep

void IREEVectorExtDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "iree-dialects/Dialect/VectorExt/IR/VectorExtAttrs.cpp.inc"
      >();
}
