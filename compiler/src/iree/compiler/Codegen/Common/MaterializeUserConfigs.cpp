// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <iterator>
#include "iree/compiler/Codegen/Common/PassDetail.h"
#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Common/UserConfig.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/iterator_range.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/Transforms/TransformInterpreterUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-codegen-materialize-user-configs"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir::iree_compiler {

llvm::cl::opt<std::string> clCodegenTransformDialectLibraryFileName(
    "iree-codegen-transform-dialect-library",
    llvm::cl::desc(
        "File path to a module containing a library of transform dialect"
        "strategies. Can be suffixed with the name of a transform sequence"
        "within the library to run as preprocessing per executable variant."
        "This is specified as <file-path>@<sequence-name>. If not specified,"
        "this will default to `__kernel_config`."),
    llvm::cl::init(""));

namespace {

static const char kTranslationInfoAttrName[] = "translation_info";

enum StrategyRunResult {
  Success = 0,
  NotFound = 1,
  Failed = 2,
};

static StrategyRunResult
runTransformConfigurationStrategy(Operation *payloadRoot,
                                  StringRef entryPointName,
                                  ModuleOp &transformLibrary) {
  /// If we have a symbol, verify the existence of the symbol within the
  /// transform library.
  Operation *entryPoint = transform::detail::findTransformEntryPoint(
      payloadRoot, transformLibrary, entryPointName);
  if (!entryPoint) {
    return StrategyRunResult::NotFound;
  }

  transform::TransformOptions options;
  if (failed(transform::applyTransformNamedSequence(
          payloadRoot, entryPoint, transformLibrary,
          options.enableExpensiveChecks(true)))) {
    return StrategyRunResult::Failed;
  }
  return StrategyRunResult::Success;
}

struct MaterializeUserConfigsPass
    : public MaterializeUserConfigsBase<MaterializeUserConfigsPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registerTransformDialectTranslationDependentDialects(registry);
  }

  void runOnOperation() override {
    IREE::HAL::ExecutableVariantOp variantOp = getOperation();
    ModuleOp moduleOp = variantOp.getInnerModule();
    llvm::StringMap<IREE::HAL::ExecutableExportOp> exportOps =
        getAllEntryPoints(moduleOp);
    MLIRContext *context = moduleOp.getContext();

    // Parse the file path and kernel config strategy from flags. There are
    // two possible usage flows for transform dialect libraries.
    //   1. Use `__kernel_config` to match and annotate variants with the
    //      strategy to use. This could either be a transform dialect strategy
    //      or any other IREE codegen pipeline.
    //
    //   2. Use the configuration strategy to do codegen directly. At the end of
    //      the strategy, the variant needs to be annotated with
    //      "translation_info" = #iree_codegen.translation_info<None>
    SmallVector<StringRef, 2> parts;
    llvm::SplitString(llvm::StringRef(clCodegenTransformDialectLibraryFileName),
                      parts, "@");
    if (parts.size() > 2) {
      variantOp.emitError()
          << "Invalid transform library path and sequence name "
          << clCodegenTransformDialectLibraryFileName;
      return signalPassFailure();
    }
    bool hasTransformLibrary = !parts.empty();

    std::string libraryFileName;
    if (hasTransformLibrary) {
      if (parts[0].empty()) {
        variantOp.emitError() << "Cannot specify an empty library path";
        return signalPassFailure();
      }
      libraryFileName = parts[0];
    }

    std::string entrySequenceName;
    // Check if the user specified a custom entry point name.
    if (parts.size() == 2) {
      if (parts[1].empty()) {
        variantOp.emitError() << "Cannot specify an empty sequence name";
        return signalPassFailure();
      }
      entrySequenceName = parts[1];
    } else {
      entrySequenceName = "__kernel_config";
    }

    LDBG("MaterializeUserConfigsPass on variant: " << variantOp);
    std::optional<ModuleOp> transformLibrary = std::nullopt;
    if (hasTransformLibrary) {
      auto dialect =
          context->getOrLoadDialect<IREE::Codegen::IREECodegenDialect>();
      auto maybeTransformLibrary =
          dialect->getOrLoadTransformLibraryModule(libraryFileName);
      if (failed(maybeTransformLibrary)) {
        variantOp.emitError()
            << "failed to load transform library module: " << libraryFileName;
        return signalPassFailure();
      }
      transformLibrary = *maybeTransformLibrary;
      LDBG("--found transform library @" << libraryFileName);

      auto runResult = runTransformConfigurationStrategy(
          variantOp, entrySequenceName, *transformLibrary);
      if (runResult == StrategyRunResult::NotFound) {
        variantOp.emitError() << "transform kernel config strategy `"
                              << entrySequenceName << " not found";
        return signalPassFailure();
      } else if (runResult == StrategyRunResult::Failed) {
        variantOp.emitError() << "transform kernel config strategy `"
                              << entrySequenceName << "` failed to apply";
        return signalPassFailure();
      }
    }

    LDBG("--start iterating over: "
         << std::distance(moduleOp.getOps<mlir::FunctionOpInterface>().begin(),
                          moduleOp.getOps<mlir::FunctionOpInterface>().end())
         << " functions");
    std::optional<IREE::Codegen::TranslationInfoAttr> translationInfo;
    for (auto funcOp : moduleOp.getOps<mlir::FunctionOpInterface>()) {
      auto exportOp = exportOps.lookup(funcOp.getName());
      if (!exportOp) {
        continue;
      }

      /// Nothing to do if the export already has a config.
      if (getTranslationInfo(exportOp)) {
        continue;
      }

      /// First, apply all user configs.
      auto res = funcOp.walk([&](Operation *op) {
        if (auto compilationInfo = getCompilationInfo(op)) {
          if (failed(setUserConfig(funcOp, op, compilationInfo))) {
            return WalkResult::interrupt();
          }
        }
        return WalkResult::advance();
      });

      if (res.wasInterrupted()) {
        moduleOp.emitOpError("error in setting user configuration");
        return signalPassFailure();
      }
    }

    LDBG("--guaranteed unique translationInfo: " << translationInfo);
    /// We only need to resolve symbols for transform dialect based strategies.
    if (!translationInfo ||
        translationInfo.value().getDispatchLoweringPassPipeline() !=
            IREE::Codegen::DispatchLoweringPassPipeline::
                TransformDialectCodegen) {
      return;
    }

    // From now on, we know we have a transform dialect strategy. We now need to
    // ensure it can resolve and apply in a subsequent interpreter pass or else
    // we need to fall back to codegen.
    bool failedToResolve = false;
    auto g = llvm::make_scope_exit([&]() {
      if (!failedToResolve)
        return;

      exportOps = getAllEntryPoints(variantOp.getInnerModule());
      for (auto &it : exportOps) {
        auto exportOp = it.second;
        if (getTranslationInfo(exportOp) == translationInfo) {
          exportOp->removeAttr(kTranslationInfoAttrName);
        }
      }
    });

    std::optional<SymbolRefAttr> strategyName =
        translationInfo.value().getCodegenSpec();
    if (!strategyName || *strategyName == SymbolRefAttr()) {
      failedToResolve = true;
      return;
    }

    /// If we have a symbol, verify the existence of the symbol within the
    /// transform library.
    StringRef entryPoint = strategyName->getLeafReference();
    if (!transformLibrary || !(*transformLibrary) ||
        !transform::detail::findTransformEntryPoint(
            variantOp, *transformLibrary, entryPoint)) {
      moduleOp.emitOpError("failed to find transform strategy symbol");
      failedToResolve = true;
    }
  }

private:
  /// Transform interpreter options.
  transform::TransformOptions options;
};

} // namespace

std::unique_ptr<OperationPass<IREE::HAL::ExecutableVariantOp>>
createMaterializeUserConfigsPass() {
  return std::make_unique<MaterializeUserConfigsPass>();
}

} // namespace mlir::iree_compiler
