// RUN: mlir-pdll %S/patterns.pdll -x=mlir | \
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

// CHECK-LABEL:   func.func @fully_connected(
// CHECK-SAME:                           %[[LHS:.*]]: tensor<100x200xi8>,
// CHECK-SAME:                           %[[RHS:.*]]: tensor<300x200xi4>) -> tensor<100x300xf32> {
// CHECK:           %{{.*}} = xnnpack.fully_connected_nc_qd8_f32_qc4w %[[LHS]], %[[RHS]], transpose_rhs = false, kernel_id = -1 : (tensor<100x200xi8>, tensor<300x200xi4>) -> tensor<100x300xf32>
func.func @fully_connected(%input : tensor<100x200xi8>, %weight : tensor<300x200xi4>) -> tensor<100x300xf32> {
  %weight_cast = stablehlo.convert %weight : (tensor<300x200xi4>) -> tensor<300x200xi8>
  %dot_general = stablehlo.dot_general %input, %weight_cast, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<100x200xi8>, tensor<300x200xi8>) -> tensor<100x300xi32>
  %out = stablehlo.convert %dot_general : (tensor<100x300xi32>) -> tensor<100x300xf32>
  return %out : tensor<100x300xf32>
}
