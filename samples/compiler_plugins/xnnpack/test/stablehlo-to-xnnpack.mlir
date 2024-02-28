// RUN: iree-opt --iree-plugin=xnnpack --iree-print-plugin-info --pass-pipeline='builtin.module(iree-stablehlo-to-xnnpack)' %s | FileCheck %s

// CHECK-LABEL:   func.func @fully_connected(
// CHECK-SAME:                           %[[LHS:.*]]: tensor<100x200xi8>,
// CHECK-SAME:                           %[[RHS:.*]]: tensor<300x200xi4>) -> tensor<100x300xf32> {
// CHECK:           %[[C8:.*]] = stablehlo.constant dense<8>
// CHECK:           %[[C8_I4:.*]] = stablehlo.convert %[[C8]] : (tensor<{{.*}}i8>) -> tensor<{{.*}}i4>
// CHECK:           %[[RHS_UI4:.*]] = stablehlo.xor %[[RHS]], %[[C8_I4]]
// CHECK:           %{{.*}} = xnnpack.fully_connected_nc_qd8_f32_qc4w %[[LHS]], %[[RHS_UI4]] transpose_rhs = false : (tensor<100x200xi8>, tensor<300x200xi4>) -> tensor<100x300xf32>
func.func @fully_connected(%input : tensor<100x200xi8>, %kernel : tensor<300x200xi4>) -> tensor<100x300xf32> {
  %kernel_cast = stablehlo.convert %kernel : (tensor<300x200xi4>) -> tensor<300x200xi8>
  %dot_general = stablehlo.dot_general %input, %kernel_cast, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<100x200xi8>, tensor<300x200xi8>) -> tensor<100x300xi32>
  %out = stablehlo.convert %dot_general : (tensor<100x300xi32>) -> tensor<100x300xf32>
  return %out : tensor<100x300xf32>
}

// CHECK-LABEL:   func.func @fully_connected$transpose(
// CHECK-SAME:                                     %[[LHS:.*]]: tensor<100x200xi8>,
// CHECK-SAME:                                     %[[RHS:.*]]: tensor<200x300xi4>) -> tensor<100x300xf32> {
// CHECK:           %[[C8:.*]] = stablehlo.constant dense<8>
// CHECK:           %[[C8_I4:.*]] = stablehlo.convert %[[C8]] : (tensor<{{.*}}xi8>) -> tensor<{{.*}}xi4>
// CHECK:           %[[RHS_UI4:.*]] = stablehlo.xor %[[RHS]], %[[C8_I4]]
// CHECK:           %{{.*}} = xnnpack.fully_connected_nc_qd8_f32_qc4w %[[LHS]], %[[RHS_UI4]] transpose_rhs = true : (tensor<100x200xi8>, tensor<200x300xi4>) -> tensor<100x300xf32>
func.func @fully_connected$transpose(%input : tensor<100x200xi8>, %kernel : tensor<200x300xi4>) -> tensor<100x300xf32> {
  %kernel_cast = stablehlo.convert %kernel : (tensor<200x300xi4>) -> tensor<200x300xi8>
  %dot_general = stablehlo.dot_general %input, %kernel_cast, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<100x200xi8>, tensor<200x300xi8>) -> tensor<100x300xi32>
  %out = stablehlo.convert %dot_general : (tensor<100x300xi32>) -> tensor<100x300xf32>
  return %out : tensor<100x300xf32>
}

// CHECK-LABEL:   func.func @fully_connected$rank3_input(
// CHECK-SAME:                           %[[LHS:.*]]: tensor<1x100x200xi8>,
// CHECK-SAME:                           %[[RHS:.*]]: tensor<300x200xi4>) -> tensor<1x100x300xf32> {
// CHECK:           %[[C8:.*]] = stablehlo.constant dense<8>
// CHECK:           %[[LHS_RESHAPE:.*]] = stablehlo.reshape %[[LHS]] : (tensor<1x100x200xi8>) -> tensor<100x200xi8>
// CHECK:           %[[C8_I4:.*]] = stablehlo.convert %[[C8]] : (tensor<{{.*}}i8>) -> tensor<{{.*}}i4>
// CHECK:           %[[RHS_UI4:.*]] = stablehlo.xor %[[RHS]], %[[C8_I4]]
// CHECK:           %[[FULLY_CONNECTED:.*]] = xnnpack.fully_connected_nc_qd8_f32_qc4w %[[LHS_RESHAPE]], %[[RHS_UI4]] transpose_rhs = false : (tensor<100x200xi8>, tensor<300x200xi4>) -> tensor<100x300xf32>
// CHECK:           %{{.*}} = stablehlo.reshape %[[FULLY_CONNECTED]] : (tensor<100x300xf32>) -> tensor<1x100x300xf32>
func.func @fully_connected$rank3_input(%input : tensor<1x100x200xi8>, %kernel : tensor<300x200xi4>) -> tensor<1x100x300xf32> {
  %kernel_cast = stablehlo.convert %kernel : (tensor<300x200xi4>) -> tensor<300x200xi8>
  %dot_general = stablehlo.dot_general %input, %kernel_cast, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x100x200xi8>, tensor<300x200xi8>) -> tensor<1x100x300xi32>
  %out = stablehlo.convert %dot_general : (tensor<1x100x300xi32>) -> tensor<1x100x300xf32>
  return %out : tensor<1x100x300xf32>
}

// CHECK-LABEL:   func.func @fully_connected$no_defining_op_for_convert_operand(
func.func @fully_connected$no_defining_op_for_convert_operand(%input : tensor<i8>) -> tensor<i16> {
  %out = stablehlo.convert %input : (tensor<i8>) -> tensor<i16>
  return %out : tensor<i16>
}

// CHECK-LABEL:   func.func @fully_connected$input_cast(
// CHECK-SAME:                           %[[LHS:.*]]: tensor<100x200xf32>
// CHECK:           %[[LHS_CAST:.*]] = stablehlo.convert %[[LHS]] : (tensor<{{.*}}f32>) -> tensor<{{.*}}i8>
// CHECK:           %{{.*}} = xnnpack.fully_connected_nc_qd8_f32_qc4w %[[LHS_CAST]]
func.func @fully_connected$input_cast(%input : tensor<100x200xf32>, %kernel : tensor<300x200xi4>) -> tensor<100x300xf32> {
  %input_cast = stablehlo.convert %input : (tensor<100x200xf32>) -> tensor<100x200xi8>
  %kernel_cast = stablehlo.convert %kernel : (tensor<300x200xi4>) -> tensor<300x200xi8>
  %dot_general = stablehlo.dot_general %input_cast, %kernel_cast, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<100x200xi8>, tensor<300x200xi8>) -> tensor<100x300xi32>
  %out = stablehlo.convert %dot_general : (tensor<100x300xi32>) -> tensor<100x300xf32>
  return %out : tensor<100x300xf32>
}
