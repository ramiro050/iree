// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "compiler/plugins/input/StableHLO/Conversion/Preprocessing/Passes.h"
#include "iree/compiler/PluginAPI/Client.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Transforms/Passes.h"
#include "xnnpack/Conversion/Passes.h"
#include "xnnpack/IR/XnnpackDialect.h"
#include "xnnpack/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::iree_compiler;

namespace {

struct XnnpackOptions {
  void bindOptions(OptionsBinder &binder) {}
};

struct XnnpackSession : public PluginSession<XnnpackSession, XnnpackOptions> {
  static void registerPasses() {
    IREE::Xnnpack::registerXnnpackPluginTransformsPasses();
    IREE::Xnnpack::registerXnnpackPluginConversionPasses();
  }

  void onRegisterDialects(DialectRegistry &registry) override {
    registry.insert<IREE::Xnnpack::XnnpackDialect>();
  }

  LogicalResult onActivate() override { return success(); }

  void extendInputConversionPreprocessingPassPipeline(
      OpPassManager &pm, InputDialectOptions::Type inputType) override {
    pm.addPass(IREE::Xnnpack::createConvertStablehloToXnnpack());
    pm.addNestedPass<func::FuncOp>(
        mlir::iree_compiler::stablehlo::createStableHLOCanonicalize());
    pm.addNestedPass<func::FuncOp>(mlir::createCSEPass());
    pm.addPass(IREE::Xnnpack::createSetIdAttributes());
  }

  void extendPreprocessingPassPipeline(OpPassManager &pm) override {
    pm.addPass(IREE::Xnnpack::createLegalizeXnnpack());
  }
};

}  // namespace

IREE_DEFINE_COMPILER_OPTION_FLAGS(XnnpackOptions);

extern "C" bool iree_register_compiler_plugin_xnnpack(
    mlir::iree_compiler::PluginRegistrar *registrar) {
  registrar->registerPlugin<XnnpackSession>("xnnpack");
  return true;
}
