// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_FLOW_TRANSFORMS_PASSES_H_
#define IREE_COMPILER_DIALECT_FLOW_TRANSFORMS_PASSES_H_

#include <functional>

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "llvm/ADT/StringMap.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"

namespace mlir::iree_compiler::IREE::Flow {

//===----------------------------------------------------------------------===//
// Pipelines
//===----------------------------------------------------------------------===//

/// This is a placeholder for future. We should pass all the options through the
/// struct.
struct TransformOptions : public PassPipelineOptions<TransformOptions> {};

// Adds a set of passes to the given pass manager that run the required flow
// transforms in the canonical order.
//
// Most translation code should prefer to use this instead of manually adding
// the passes themselves to ensure that expected pass ordering is observed.
//
// The expected usage is:
//   Input legalization by one of:
//     - Directly passing supported flow plus core ops
//   buildFlowTransformPassPipeline
//   <run conversion from flow to sequencer/hal/vm/etc>
void buildFlowTransformPassPipeline(OpPassManager &passManager,
                                    const TransformOptions &transformOptions);

void registerFlowTransformPassPipeline();

//===----------------------------------------------------------------------===//
// Input canonicalization and legalization
//===----------------------------------------------------------------------===//

// Cleans up any remaining shape metadata ops after lowering.
std::unique_ptr<Pass> createCleanupTensorShapesPass();

// Creates a pass to convert dispatch.region ops to dispatch.workgroups ops.
std::unique_ptr<Pass> createConvertRegionToWorkgroupsPass();

// Pass to convert a tensor.pad operation into a linalg.fill +
// tensor.insert_slice.
std::unique_ptr<Pass>
createTensorPadToTensorInsertSlicePass(bool skipSingleLinalgOpUses = false);

// Create a pass that imports upstream patterns to fold unit extent dims
// but with IREE control.
std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createFoldUnitExtentDimsPass();

// Creates a pass to fuse Linalg operations on tensors.
std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createFusionOfTensorOpsPass(bool fuseMultiUse = false,
                            unsigned multiUseFusionIteration = 2);

// Create a pass to initialize all empty tensors after dispatch formation to
// zero or uninitialized allocations.
std::unique_ptr<Pass> createInitializeEmptyTensorsPass(bool zeroFill = false);

// Create a pass to interchange generic ops to force the reduction loop to be
// the most inner loops.
std::unique_ptr<Pass> createInterchangeGenericOpsPass();

// Create a pass to interchange generic ops to make the input indexing map
// identity.
std::unique_ptr<Pass> createInterchangeTransposeGenericOpsPass();

// Create a pass to convert operations to `flow` ops. This pass is currently
// only used for testing, since the conversion to Flow ops happens within
// dispatch region formation.
std::unique_ptr<Pass> createConvertToFlowPass();

// Decomposes top-level SCF operations to CFG.
std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createTopLevelSCFToCFGPass();

// Verifies that the input to the Flow transformation pipeline is legal.
// This includes checking for operations from dialects that are expected
// to be legalized before this pass.
std::unique_ptr<Pass> createVerifyInputLegalityPass();

//===----------------------------------------------------------------------===//
// Dispatches (flow.dispatch.region)
//===----------------------------------------------------------------------===//

// Pass to form dispatch.region ops from Linalg on tensor ops. A dispatch region
// is created for each tiled loop nest. This pass only moves the root compute op
// into the dispatch region, allowing producers to be outside.
/// This struct is the same struct that is auto-generated from tablegen file for
/// the pass definition. THis is manually copied from the `Passes.h.inc` file
/// generated below.
// TODO(ravishankarm): Move the passes in Flow to use the auto-generated options
// struct.
struct FormDispatchRegionsOptions {
  bool fuseMultiUse = false;
  bool generateWorkloadRegion = true;
  bool fusePadWithConsumers = false;
  bool fusePadWithProducers = false;
};
std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createFormDispatchRegionsPass(FormDispatchRegionsOptions options = {});

// Pass to create `flow.dispatch.region`s for scalar computations.
std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createFormScalarDispatchesPass();

// Pass to collapse dimensions of Linalg Ops on tensor ops.
std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createCollapseDimensionsPass();

// Pass to clone into dispatch regions producers of values used in the dispatch
// regions but defined in the above. This prepares the dispatch regions for
// converting to dispatch workgroups with explicit captures.
std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createCloneProducersIntoDispatchRegionsPass();

//===----------------------------------------------------------------------===//
// Dispatches (flow.dispatch.workgroups)
//===----------------------------------------------------------------------===//

// Pass to perform dispatch of dispatch.region ops that contain Linalg on tensor
// ops by tiling and distribution. A dispatch region is created for each tiled
// loop nest.
std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createFormDispatchWorkgroupsPass(bool generateWorkloadRegion = true);

// Pass to perform dispatch of Linalg on tensor ops by using the transform
// dialect. Dispatch regions are created as specified by the transform module
// that is parsed from `transformFileName`.
std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createDispatchWithTransformDialect(
    llvm::StringRef transformFileName = llvm::StringRef(),
    llvm::StringRef debugPayloadRootTag = llvm::StringRef(),
    llvm::StringRef debugTransformRootTag = llvm::StringRef());

// Captures dynamic shape dimensions required by dispatch operands.
std::unique_ptr<Pass> createCaptureDispatchDynamicDimsPass();

// Outlines external dispatches into executables.
std::unique_ptr<OperationPass<mlir::ModuleOp>>
createOutlineDispatchExternsPass();

// Outlines dispatch regions into executables.
std::unique_ptr<OperationPass<mlir::ModuleOp>>
createOutlineDispatchRegionsPass();

// Annotates executable dispatches based on their contents.
std::unique_ptr<OperationPass<mlir::ModuleOp>> createAnnotateDispatchesPass();

// Injects tracing markers for dispatch operation tensor inputs and outputs.
std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createInjectDispatchTracingPass();

// Crops the program and inserts trace markers at the specified symbols.
std::unique_ptr<OperationPass<mlir::ModuleOp>>
createInsertDebugTargetAtSymbolPass(std::string breakDebugTarget = "",
                                    std::string traceDebugTarget = "");

// Crops the program and inserts trace markers at the specified ordinals.
std::unique_ptr<OperationPass<mlir::ModuleOp>>
createInsertDebugTargetAtOrdinalPass(std::string breakDebugTarget = "",
                                     std::string traceDebugTarget = "");

// Exports all functions and dispatch executables as `() -> ()` benchmark funcs.
std::unique_ptr<OperationPass<mlir::ModuleOp>> createExportBenchmarkFuncsPass();

//===----------------------------------------------------------------------===//
// Optimizations
//===----------------------------------------------------------------------===//

// Outlines large tensor constants into util.globals at the module level.
std::unique_ptr<OperationPass<mlir::ModuleOp>>
createOutlineLargeConstantsPass();

// Deduplicates equivalent executables.
std::unique_ptr<OperationPass<mlir::ModuleOp>>
createDeduplicateExecutablesPass();

// Create a pass to split reduction dimension.
std::unique_ptr<Pass> createSplitReductionPass();

// Create a pass to collapse reduction dimensions
std::unique_ptr<Pass> createCollapseDimsPass();

//===----------------------------------------------------------------------===//
// Module Analysis and Finalization
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Simplification and Development Tools
//===----------------------------------------------------------------------===//

/// Creates a pass to dump a graph for dispatches
std::unique_ptr<Pass>
createDumpDispatchGraphPass(raw_ostream &os = llvm::errs());

//===----------------------------------------------------------------------===//
// Register all Passes
//===----------------------------------------------------------------------===//

void registerFlowPasses();

} // namespace mlir::iree_compiler::IREE::Flow

#endif // IREE_COMPILER_DIALECT_FLOW_TRANSFORMS_PASSES_H_
