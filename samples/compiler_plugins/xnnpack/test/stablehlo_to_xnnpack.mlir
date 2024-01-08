// RUN: iree-opt --iree-plugin=xnnpack_sample --iree-print-plugin-info --pass-pipeline='builtin.module(iree-stablehlo-to-xnnpack)' %s | FileCheck %s

// CHECK-LABEL:   func.func @dot_general(
// CHECK:           %{{.*}} = xnnpack.batch_matrix_multiply
func.func @dot_general(%a : tensor<1x100x200xi8>, %b : tensor<200x300xi8>) -> tensor<1x100x300xi32> {
  %out = stablehlo.dot_general %a, %b, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x100x200xi8>, tensor<200x300xi8>) -> tensor<1x100x300xi32>
  func.return %out : tensor<1x100x300xi32>
}

