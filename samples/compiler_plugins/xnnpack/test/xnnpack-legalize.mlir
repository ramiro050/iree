// RUN: iree-opt --iree-plugin=xnnpack_sample --iree-print-plugin-info --pass-pipeline='builtin.module(iree-xnnpack-legalize)' %s | FileCheck %s

// CHECK-LABEL:   func.func private @xnnpack.mul2(
// CHECK-SAME:                                    %[[VAL_0:.*]]: tensor<?xf32>,
// CHECK-SAME:                                    %[[VAL_1:.*]]: tensor<?xf32>) -> tensor<?xf32> {
// CHECK:           %[[VAL_2:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_3:.*]] = tensor.dim %[[VAL_0]], %[[VAL_2]] : tensor<?xf32>
// CHECK:           %[[VAL_4:.*]] = tensor.empty(%[[VAL_3]]) : tensor<?xf32>
// CHECK:           %[[VAL_5:.*]] = flow.dispatch.region -> (tensor<?xf32>{%[[VAL_3]]}) {
// CHECK:             %[[VAL_6:.*]] = iree_codegen.ukernel.generic "xnnpack.mul2_workgroup" ins(%[[VAL_0]], %[[VAL_1]] : tensor<?xf32>, tensor<?xf32>) outs(%[[VAL_4]] : tensor<?xf32>) (%[[VAL_3]] : index) -> tensor<?xf32>
// CHECK:             flow.return %[[VAL_6]] : tensor<?xf32>
// CHECK:           return %[[VAL_5]] : tensor<?xf32>

// CHECK-LABEL:   func.func @main(
// CHECK:           %[[VAL_2:.*]] = call @xnnpack.mul2(%[[VAL_0]], %[[VAL_1]])
func.func @main(%a : tensor<?xf32>, %b : tensor<?xf32>) -> tensor<?xf32> {
  %c = xnnpack.mul2 %a, %b : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  func.return %c : tensor<?xf32>
}
