// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_SAMPLES_COMPILER_PLUGINS_XNNPACK_IR_XNNPACKOPS_H_
#define IREE_SAMPLES_COMPILER_PLUGINS_XNNPACK_IR_XNNPACKOPS_H_

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"

// Include generated.
#define GET_OP_CLASSES
#include "xnnpack/IR/XnnpackOps.h.inc"  // IWYU pragma: keep

#endif  // IREE_SAMPLES_COMPILER_PLUGINS_XNNPACK_IR_XNNPACKOPS_H_
