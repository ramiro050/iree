// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef XNNPACK_SAMPLE
#define XNNPACK_SAMPLE

include "mlir/IR/OpBase.td"

def Xnnpack_Dialect : Dialect {
  let name = "xnnpack";
  let cppNamespace = "::mlir::iree_compiler::IREE::Xnnpack";
}

class Xnnpack_Op<string mnemonic, list<Trait> traits = []> :
    Op<Xnnpack_Dialect, mnemonic, traits> {
}

// TODO(ramiro050): remove this op. This is from simple io example
def Xnnpack_PrintOp : Xnnpack_Op<"print", []> {
  let summary = [{Print}];
  let arguments = (ins);
  let results = (outs);

  let assemblyFormat = [{
    attr-dict
  }];
}

def Xnnpack_Mul2Op : Xnnpack_Op<"mul2", []> {
  let summary = "Elementwise multiplication";
  let arguments = (ins AnyRankedTensor:$a,
                       AnyRankedTensor:$b);
  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = [{
    $a `,` $b attr-dict `:` functional-type(operands, results)
  }];
}

#endif
