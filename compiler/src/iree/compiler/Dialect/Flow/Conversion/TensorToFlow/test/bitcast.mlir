// RUN: iree-opt --allow-unregistered-dialect --split-input-file --iree-flow-convert-to-flow %s | FileCheck %s

 util.func public @static_tensor_bitcast(%arg0: tensor<4x4xf32>) -> tensor<4x4xi32> {
  // CHECK-DAG: %[[RESULT:.*]] = flow.tensor.bitcast %arg0 : tensor<4x4xf32> -> tensor<4x4xi32>
  // CHECK: util.return %[[RESULT]]
  %0 = tensor.bitcast %arg0 : tensor<4x4xf32> to tensor<4x4xi32>
  util.return %0 : tensor<4x4xi32>
}

// -----

 util.func public @dynamic_tensor_bitcast(%arg0: tensor<?x?xf32>) -> tensor<?x?xi32> {
  // CHECK: %[[DIM0:.+]] = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  // CHECK: %[[DIM1:.+]] = tensor.dim %arg0, %c1 : tensor<?x?xf32>
  // CHECK: %[[RESULT:.+]] = flow.tensor.bitcast %arg0 : tensor<?x?xf32>{%[[DIM0]], %[[DIM1]]} -> tensor<?x?xi32>{%[[DIM0]], %[[DIM1]]}
  // CHECK: util.return %[[RESULT]]
  %0 = tensor.bitcast %arg0 : tensor<?x?xf32> to tensor<?x?xi32>
  util.return %0 : tensor<?x?xi32>
}
