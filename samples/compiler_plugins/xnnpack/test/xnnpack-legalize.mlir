// RUN: iree-opt --iree-plugin=xnnpack --iree-print-plugin-info --pass-pipeline='builtin.module(iree-xnnpack-legalize)' --split-input-file %s | FileCheck %s

// CHECK-LABEL:   func.func private @xnnpack.multiply2_0(
// CHECK-SAME:                                    %[[A:.*]]: tensor<?xf32>,
// CHECK-SAME:                                    %[[B:.*]]: tensor<?xf32>) -> tensor<?xf32> {
// CHECK:           %[[OUT:.*]] = tensor.empty(%[[A_DIM:.*]]) : tensor<?xf32>
// CHECK:           %[[RESULT:.*]] = flow.dispatch.region -> (tensor<?xf32>{%[[A_DIM]]}) {
// CHECK:             %[[UKERNEL:.*]] = iree_codegen.ukernel.generic "xnnpack.multiply2_workgroup" ins(%[[A]], %[[B]] : tensor<?xf32>, tensor<?xf32>) outs(%[[OUT]] : tensor<?xf32>) (%[[A_DIM]], %[[B_DIM:.*]], %[[A_DIM]] : index, index, index) -> tensor<?xf32>
// CHECK:             flow.return %[[UKERNEL]] : tensor<?xf32>
// CHECK:           } count()
// CHECK:             %[[ONE:.*]] = arith.constant 1 : index
// CHECK:             flow.return %[[ONE]], %[[ONE]], %[[ONE]]
// CHECK:           return %[[RESULT]] : tensor<?xf32>

// CHECK-LABEL:   func.func @multiply2(
// CHECK:           %{{.*}} = call @xnnpack.multiply2_0
func.func @multiply2(%a : tensor<?xf32>, %b : tensor<?xf32>) -> tensor<?xf32> {
  %c = xnnpack.multiply2 %a, %b : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  func.return %c : tensor<?xf32>
}

// -----

// CHECK-LABEL:   func.func private @xnnpack.batch_matrix_multiply_0(
// CHECK-SAME:                                      %[[A:.*]]: tensor<?x?x?xf32>,
// CHECK-SAME:                                      %[[B:.*]]: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
// CHECK:           %[[OUT:.*]] = tensor.empty(%[[A_DIM_0:.*]], %[[A_DIM_1:.*]], %[[B_DIM_2:.*]]) : tensor<?x?x?xf32>
// CHECK:           %[[RESULT:.*]] = flow.dispatch.region -> (tensor<?x?x?xf32>{%[[A_DIM_0]], %[[A_DIM_1]], %[[B_DIM_2]]}) {
// CHECK:             %[[UKERNEL:.*]] = iree_codegen.ukernel.generic "xnnpack.batch_matrix_multiply_workgroup" ins(%[[A]], %[[B]] : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%[[OUT]] : tensor<?x?x?xf32>) (%[[A_DIM_0]], %[[A_DIM_1]], %[[A_DIM_2:.*]], %[[B_DIM_0:.*]], %[[B_DIM_1:.*]], %[[B_DIM_2]], %[[A_DIM_0]], %[[A_DIM_1]], %[[B_DIM_2]] : index, index, index, index, index, index, index, index, index) -> tensor<?x?x?xf32>
// CHECK:             flow.return %[[UKERNEL]] : tensor<?x?x?xf32>
// CHECK:           } count()
// CHECK:             %[[ONE:.*]] = arith.constant 1 : index
// CHECK:             flow.return %[[ONE]], %[[ONE]], %[[ONE]]
// CHECK:           }
// CHECK:           return %[[RESULT]] : tensor<?x?x?xf32>

// CHECK-LABEL:   func.func @batch_matrix_multiply(
// CHECK:           %{{.*}} = call @xnnpack.batch_matrix_multiply_0
func.func @batch_matrix_multiply(%a : tensor<?x?x?xf32>, %b : tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
  %c = xnnpack.batch_matrix_multiply %a, %b : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  func.return %c : tensor<?x?x?xf32>
}

// -----

// CHECK-LABEL:   func.func private @xnnpack.fully_connected_nc_qd8_f32_qc4w_0(
// CHECK-SAME:                                                               %[[A:.*]]: tensor<?x?xi8>,
// CHECK-SAME:                                                               %[[B:.*]]: tensor<?x?xi4>) -> tensor<?x?xf32> {
// CHECK:           %[[OUT:.*]] = tensor.empty(%[[A_DIM_0:.*]], %[[B_DIM_0:.*]]) : tensor<?x?xf32>
// CHECK:           %[[RESULT:.*]] = flow.dispatch.region -> (tensor<?x?xf32>{%[[A_DIM_0]], %[[B_DIM_0]]}) {
// CHECK:             %[[UKERNEL:.*]] = iree_codegen.ukernel.generic "xnnpack.fully_connected_nc_qd8_f32_qc4w_workgroup" ins(%[[A]], %[[B]] : tensor<?x?xi8>, tensor<?x?xi4>) outs(%[[OUT]] : tensor<?x?xf32>) (%[[A_DIM_0]], %[[A_DIM_1:.*]], %[[B_DIM_0]], %[[B_DIM_1:.*]], %[[A_DIM_0]], %[[B_DIM_0]], %[[TRANSPOSE_RHS:.*]], %[[KERNEL_ID:.*]] : index, index, index, index, index, index, i8, index) -> tensor<?x?xf32>
// CHECK:             flow.return %[[UKERNEL]] : tensor<?x?xf32>
// CHECK:           } count() -> (index, index, index) {
// CHECK:             %[[ONE:.*]] = arith.constant 1 : index
// CHECK:             flow.return %[[ONE]], %[[ONE]], %[[ONE]] : index, index, index
// CHECK:           }
// CHECK:           return %[[RESULT]] : tensor<?x?xf32>

// CHECK-LABEL:   func.func @fully_connected(
// CHECK:           %{{.*}} = call @xnnpack.fully_connected_nc_qd8_f32_qc4w_0
func.func @fully_connected(%a : tensor<?x?xi8>, %b : tensor<?x?xi4>) -> tensor<?x?xf32> {
  %c = xnnpack.fully_connected_nc_qd8_f32_qc4w %a, %b, transpose_rhs = false, kernel_id = -1 : (tensor<?x?xi8>, tensor<?x?xi4>) -> tensor<?x?xf32>
  func.return %c : tensor<?x?xf32>
}

// -----

// CHECK-LABEL:   func.func private @xnnpack.fully_connected_nc_qd8_f32_qc4w_0(
// CHECK-LABEL:   func.func private @xnnpack.fully_connected_nc_qd8_f32_qc4w_1(

// CHECK-LABEL:   func.func @fully_connected$multiple_static(
// CHECK:           %{{.*}} = call @xnnpack.fully_connected_nc_qd8_f32_qc4w_0
// CHECK:           %{{.*}} = call @xnnpack.fully_connected_nc_qd8_f32_qc4w_1
func.func @fully_connected$multiple_static(%input_0 : tensor<2x8xi8>, %kernel_0 : tensor<4x8xi4>,
                                           %input_1 : tensor<2x16xi8>, %kernel_1 : tensor<4x16xi4>)
                                           -> (tensor<2x4xf32>, tensor<2x4xf32>) {
  %output_0 = xnnpack.fully_connected_nc_qd8_f32_qc4w %input_0, %kernel_0, transpose_rhs = false, kernel_id = -1 : (tensor<2x8xi8>, tensor<4x8xi4>) -> tensor<2x4xf32>
  %output_1 = xnnpack.fully_connected_nc_qd8_f32_qc4w %input_1, %kernel_1, transpose_rhs = false, kernel_id = -1 : (tensor<2x16xi8>, tensor<4x16xi4>) -> tensor<2x4xf32>
  func.return %output_0, %output_1 : tensor<2x4xf32>, tensor<2x4xf32>
}
