// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_GLOBALOPTIMIZATION_PASSES_H_
#define IREE_COMPILER_GLOBALOPTIMIZATION_PASSES_H_

#include <functional>

#include "iree/compiler/Pipelines/Options.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

namespace mlir::iree_compiler::GlobalOptimization {

// We have a layer of indirection around the GlobalOptimizationOptions because
// we also need a reference to the const-eval builder, which is injected
// in by callers.
struct TransformOptions : public PassPipelineOptions<TransformOptions> {
  GlobalOptimizationOptions options;

  // Hook to populate a constant evaluation pass pipeline. If nullptr, then
  // no passes are added for constant evaluation. This must be injected in
  // because constant-evaluators can depend on the whole compiler, of which
  // this is a part, and we maintain strict optionality for this component.
  std::function<void(OpPassManager &passManager)> buildConstEvalPassPipeline;
};

// Subset of the overall pass pipeline for optimizing globals and numerics.
// We may ultimately break this out separately so creating a syntactic
// distinction to keep that as an option.
void buildGlobalOptimizationPassPipeline(
    OpPassManager &mainPassManager, const TransformOptions &transformOptions);

//===----------------------------------------------------------------------===//
// Input canonicalization and legalization
//===----------------------------------------------------------------------===//

// Cleans up any numeric narrowing ops inserted by
// iree-global-opt-infer-numeric-narrowing.
std::unique_ptr<Pass> createCleanupNumericNarrowingPass();

// Creates a pass to convert linalg convolution ops with 1x1 kernels into
// linalg.matmul
std::unique_ptr<Pass> createConvert1X1FilterConv2DToMatmulPass();

// A pass to fuse dequantization and matmul linalg.generic ops
std::unique_ptr<Pass>
createDecomposeConcatPass(bool enableConcatTransposition = false);

// Create a pass to detach elementwise ops from named Linalg ops.
std::unique_ptr<Pass> createDetachElementwiseFromNamedOpsPass();

// Apply patterns to erase unused linalg operands and remove dead code
// associated.
std::unique_ptr<OperationPass<mlir::ModuleOp>>
createEraseUnusedLinalgOperands();

// Expands tensor shape dimensions into SSA values across the program.
std::unique_ptr<OperationPass<mlir::ModuleOp>> createExpandTensorShapesPass();

// A pass to fuse dequantization and matmul linalg.generic ops
std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createFuseDequantizationMatmulPass(
    bool enableQuantizedMatmulReassociation = false);

// A pass to fuse two matmul ops and a linalg.generic Silu op
std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createFuseSiluHorizontalMatmulPass();

// Create a pass that generalizes some named Linalg ops into `linalg.generic`
// operations since the IREE compiler can handle that better.
std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createGeneralizeLinalgNamedOpsPass();

// Infers and inserts util.numeric.optional_narrow ops at points that may be
// beneficial.
std::unique_ptr<Pass> createInferNumericNarrowingPass();

// Materializes logical encodings to physical encodings if there is a single
// device target.
std::unique_ptr<OperationPass<mlir::ModuleOp>>
createMaterializeHomogeneousEncodingsPass();

// Optimizes numerics given annotations added via
// iree-global-opt-infer-numeric-narrowing.
std::unique_ptr<Pass> createOptimizeNumericsPass();

// Create a pass to raise sequence of ops to higher level linalg.ext
// representation.
std::unique_ptr<Pass> createRaiseSpecialOps();

// Removes tensors that have 0-extents.
std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createRemoveZeroExtentTensorsPass();

// Sets encoding for tensors to allow tiled execution of operations.
std::unique_ptr<Pass> createSetEncodingPass();

// Convert linalg.generic ops to linalg.batch_matmul, possibly with transposes
// on operands/result.
std::unique_ptr<Pass> createLiftGenericToTransposeBatchMatmulPass();

void registerGlobalOptimizationPipeline();

} // namespace mlir::iree_compiler::GlobalOptimization

#endif // IREE_COMPILER_GLOBALOPTIMIZATION_PASSES_H_
