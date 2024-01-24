// RUN: iree-opt --iree-plugin=xnnpack --iree-print-plugin-info --pass-pipeline='builtin.module(iree-stablehlo-to-xnnpack)' %s | FileCheck %s

// CHECK-LABEL:   func.func @fully_connected(
// CHECK-SAME:                           %[[LHS:.*]]: tensor<1x100x200xi8>,
// CHECK-SAME:                           %[[RHS:.*]]: tensor<300x200xi4>) -> tensor<1x100x300xf32> {
// CHECK:           %{{.*}} = xnnpack.fully_connected_nc_qd8_f32_qc4w %[[LHS]], %[[RHS]] : (tensor<1x100x200xi8>, tensor<300x200xi4>) -> tensor<1x100x300xf32>
func.func @fully_connected(%input : tensor<1x100x200xi8>, %kernel : tensor<300x200xi4>) -> tensor<1x100x300xf32> {
  %kernel_cast = stablehlo.convert %kernel : (tensor<300x200xi4>) -> tensor<300x200xi8>
  %dot_general = stablehlo.dot_general %input, %kernel_cast, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x100x200xi8>, tensor<300x200xi8>) -> tensor<1x100x300xi32>
  %out = stablehlo.convert %dot_general : (tensor<1x100x300xi32>) -> tensor<1x100x300xf32>
  return %out : tensor<1x100x300xf32>
}

// CHECK-LABEL:   func.func @fully_connected$transpose(
// CHECK-SAME:                                     %[[LHS:.*]]: tensor<1x100x200xi8>,
// CHECK-SAME:                                     %[[RHS:.*]]: tensor<200x300xi4>) -> tensor<1x100x300xf32> {
// CHECK:           %[[TRANSPOSE:.*]] = stablehlo.transpose %[[RHS]], dims = [1, 0] : (tensor<200x300xi4>) -> tensor<300x200xi4>
// CHECK:           %{{.*}} = xnnpack.fully_connected_nc_qd8_f32_qc4w %[[LHS]], %[[TRANSPOSE]] : (tensor<1x100x200xi8>, tensor<300x200xi4>) -> tensor<1x100x300xf32>
func.func @fully_connected$transpose(%input : tensor<1x100x200xi8>, %kernel : tensor<200x300xi4>) -> tensor<1x100x300xf32> {
  %kernel_cast = stablehlo.convert %kernel : (tensor<200x300xi4>) -> tensor<200x300xi8>
  %dot_general = stablehlo.dot_general %input, %kernel_cast, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x100x200xi8>, tensor<200x300xi8>) -> tensor<1x100x300xi32>
  %out = stablehlo.convert %dot_general : (tensor<1x100x300xi32>) -> tensor<1x100x300xf32>
  return %out : tensor<1x100x300xf32>
}