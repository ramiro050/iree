// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef XNNPACK
#define XNNPACK

include "mlir/IR/OpBase.td"

def Xnnpack_Dialect : Dialect {
  let name = "xnnpack";
  let cppNamespace = "::mlir::iree_compiler::IREE::Xnnpack";
}

class Xnnpack_Op<string mnemonic, list<Trait> traits = []> :
    Op<Xnnpack_Dialect, mnemonic, traits> {
}

def Xnnpack_Multiply2Op : Xnnpack_Op<"multiply2", []> {
  let summary = "Elementwise multiplication";
  let arguments = (ins AnyRankedTensor:$a,
                       AnyRankedTensor:$b);
  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = [{
    $a `,` $b attr-dict `:` functional-type(operands, results)
  }];
}

def Xnnpack_BatchMatrixMultiplyOp : Xnnpack_Op<"batch_matrix_multiply", []> {
  let summary = "Batch matrix multiplication";
  let arguments = (ins AnyRankedTensor:$a,
                       AnyRankedTensor:$b);
  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = [{
    $a `,` $b attr-dict `:` functional-type(operands, results)
  }];
}

#endif
