// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/PluginAPI/Client.h"
#include "llvm/Support/CommandLine.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "xnnpack/Conversion/Passes.h"
#include "xnnpack/IR/XnnpackDialect.h"
#include "xnnpack/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::iree_compiler;

namespace {

struct XnnpackOptions {
  size_t xnnpackThreads = 1;

  void bindOptions(OptionsBinder &binder) {
    static llvm::cl::OptionCategory category("XNNPACK Plugin");
    binder.opt<size_t>(
        "xnnpack-threads", xnnpackThreads,
        llvm::cl::desc("Number of threads in XNNPACK threadpool."),
        llvm::cl::cat(category));
  }
};

struct MySession : public PluginSession<MySession, XnnpackOptions> {
  static void registerPasses() {
    IREE::Xnnpack::registerXnnpackPluginTransformsPasses();
    IREE::Xnnpack::registerXnnpackPluginConversionPasses();
  }

  void onRegisterDialects(DialectRegistry &registry) override {
    registry.insert<IREE::Xnnpack::XnnpackDialect>();
  }

  LogicalResult onActivate() override { return success(); }

  void extendPreprocessingPassPipeline(OpPassManager &pm) override {
    IREE::Xnnpack::LegalizeXnnpackOptions legalizeXnnpackOptions;
    legalizeXnnpackOptions.xnnpackThreads = options.xnnpackThreads;

    pm.addPass(IREE::Xnnpack::createConvertStablehloToXnnpackPass());
    pm.addPass(IREE::Xnnpack::createLegalizeXnnpack(legalizeXnnpackOptions));
  }
};

}  // namespace

IREE_DEFINE_COMPILER_OPTION_FLAGS(XnnpackOptions);

extern "C" bool iree_register_compiler_plugin_xnnpack(
    mlir::iree_compiler::PluginRegistrar *registrar) {
  registrar->registerPlugin<MySession>("xnnpack");
  return true;
}
