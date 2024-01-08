// RUN: iree-opt --iree-plugin=xnnpack --iree-print-plugin-info --pass-pipeline='builtin.module(iree-xnnpack-legalize)' %s | FileCheck %s

// CHECK-LABEL:   func.func private @xnnpack.multiply2(
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

// CHECK-LABEL:   func.func private @xnnpack.batch_matrix_multiply(
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

// CHECK-LABEL:   func.func @multiply2(
// CHECK:           %[[VAL_2:.*]] = call @xnnpack.multiply2
func.func @multiply2(%a : tensor<?xf32>, %b : tensor<?xf32>) -> tensor<?xf32> {
  %c = xnnpack.multiply2 %a, %b : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  func.return %c : tensor<?xf32>
}

// CHECK-LABEL:   func.func @batch_matrix_multiply(
// CHECK:           %[[VAL_2:.*]] = call @xnnpack.batch_matrix_multiply
func.func @batch_matrix_multiply(%a : tensor<?x?x?xf32>, %b : tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
  %c = xnnpack.batch_matrix_multiply %a, %b : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  func.return %c : tensor<?x?x?xf32>
}
