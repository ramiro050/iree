// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/PluginAPI/Client.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "xnnpack_sample/Conversion/Passes.h"
#include "xnnpack_sample/IR/XnnpackDialect.h"
#include "xnnpack_sample/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::iree_compiler;

namespace {

struct MyOptions {
  void bindOptions(OptionsBinder &binder) {}
};

struct MySession : public PluginSession<MySession, MyOptions> {
  static void registerPasses() {
    IREE::Xnnpack::registerXnnpackPluginTransformsPasses();
    IREE::Xnnpack::registerXnnpackPluginConversionPasses();
  }

  void onRegisterDialects(DialectRegistry &registry) override {
    registry.insert<IREE::Xnnpack::XnnpackDialect>();
  }

  LogicalResult onActivate() override { return success(); }

  void extendPreprocessingPassPipeline(OpPassManager &pm) override {
    pm.addPass(IREE::Xnnpack::createConvertStablehloToXnnpackPass());
    pm.addPass(IREE::Xnnpack::createLegalizeXnnpackPass());
  }
};

}  // namespace

IREE_DEFINE_COMPILER_OPTION_FLAGS(MyOptions);

extern "C" bool iree_register_compiler_plugin_xnnpack_sample(
    mlir::iree_compiler::PluginRegistrar *registrar) {
  registrar->registerPlugin<MySession>("xnnpack_sample");
  return true;
}
