// RUN: iree-opt --iree-plugin=xnnpack --iree-print-plugin-info --pass-pipeline='builtin.module(iree-stablehlo-to-xnnpack)' %s | FileCheck %s

// CHECK-LABEL:   func.func @dot_general(
// CHECK-SAME:                           %[[LHS:.*]]: tensor<1x100x200xi8>,
// CHECK-SAME:                           %[[RHS:.*]]: tensor<200x300xi8>) -> tensor<1x100x300xi32> {
// CHECK:           %[[LHS_BROADCAST:.*]] = stablehlo.broadcast_in_dim %[[LHS]], dims = [0, 1, 2] : (tensor<1x100x200xi8>) -> tensor<1x100x200xi8>
// CHECK:           %[[RHS_BROADCAST:.*]] = stablehlo.broadcast_in_dim %[[RHS]], dims = [1, 2] : (tensor<200x300xi8>) -> tensor<1x200x300xi8>
// CHECK:           %{{.*}} = xnnpack.batch_matrix_multiply %[[LHS_BROADCAST]], %[[RHS_BROADCAST]] : (tensor<1x100x200xi8>, tensor<1x200x300xi8>) -> tensor<1x100x300xi32>
func.func @dot_general(%a : tensor<1x100x200xi8>, %b : tensor<200x300xi8>) -> tensor<1x100x300xi32> {
  %out = stablehlo.dot_general %a, %b, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x100x200xi8>, tensor<200x300xi8>) -> tensor<1x100x300xi32>
  func.return %out : tensor<1x100x300xi32>
}

