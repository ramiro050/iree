// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_HAL_TARGET_LLVMCPU_BUILTINS_UKERNEL_H_
#define IREE_COMPILER_DIALECT_HAL_TARGET_LLVMCPU_BUILTINS_UKERNEL_H_

#include "llvm/IR/Module.h"
#include "llvm/Target/TargetMachine.h"

namespace mlir::iree_compiler::IREE::HAL {

llvm::Expected<std::unique_ptr<llvm::Module>>
loadUKernelBaseBitcode(llvm::TargetMachine *targetMachine,
                       llvm::LLVMContext &context);

llvm::Expected<std::unique_ptr<llvm::Module>>
loadUKernelArchEntryPointsBitcode(llvm::TargetMachine *targetMachine,
                                  llvm::LLVMContext &context);

llvm::Expected<std::unique_ptr<llvm::Module>>
loadUKernelArchBitcode(llvm::TargetMachine *targetMachine,
                       llvm::LLVMContext &context);

} // namespace mlir::iree_compiler::IREE::HAL

#endif // IREE_COMPILER_DIALECT_HAL_TARGET_LLVMCPU_BUILTINS_UKERNEL_H_
