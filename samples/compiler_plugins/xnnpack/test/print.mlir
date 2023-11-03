// RUN: iree-opt --iree-plugin=xnnpack_sample --iree-print-plugin-info --pass-pipeline='builtin.module(iree-xnnpack-legalize)' %s | FileCheck %s

// CHECK: func.func private @xnnpack.print()
func.func @main(%a : tensor<?xf32>, %b : tensor<?xf32>) -> tensor<?xf32> {
  // CHECK: call @xnnpack.print
  xnnpack.print
  // CHECK: xnnpack.mul2
  %c = xnnpack.mul2 %a, %b : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  func.return %c : tensor<?xf32>
}
