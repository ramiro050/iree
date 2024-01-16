// RUN: mlir-pdll %S/patterns_pdll.mlir -x=mlir | \
// RUN: mlir-opt --convert-pdl-to-pdl-interp | \
// RUN: iree-opt --iree-plugin=xnnpack --iree-print-plugin-info --xnnpack-pattern-file=- \
// RUN:          --pass-pipeline='builtin.module(iree-stablehlo-to-xnnpack)' %s | \
// RUN: FileCheck %s


// CHECK-LABEL:   func.func @multiply(
// CHECK:           %{{.*}} = xnnpack.multiply2
func.func @multiply(%a : tensor<100x200xi8>, %b : tensor<100x200xi8>) -> tensor<100x200xi8> {
  %out = stablehlo.multiply %a, %b : tensor<100x200xi8>
  func.return %out : tensor<100x200xi8>
}

