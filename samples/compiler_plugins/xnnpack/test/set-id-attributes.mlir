// RUN: iree-opt --iree-plugin=xnnpack --iree-print-plugin-info --pass-pipeline='builtin.module(iree-xnnpack-set-id-attributes)' --split-input-file %s | FileCheck %s

// CHECK-LABEL:   func.func @fully_connected
// CHECK: {{.*}} xnnpack.fully_connected_nc_qd8_f32_qc4w {{.*}} kernel_id = 0
func.func @fully_connected(%input : tensor<2x4xi8>) -> tensor<2x6xf32> {
  %kernel = stablehlo.constant dense<0> : tensor<6x4xi4>
  %c8 = stablehlo.constant dense<8> : tensor<6x4xi8>
  %c8_int4 = stablehlo.convert %c8 : (tensor<6x4xi8>) -> tensor<6x4xi4>
  %xor = stablehlo.xor %kernel, %c8_int4 : (tensor<6x4xi4>, tensor<6x4xi4>) -> tensor<6x4xi4>
  %out = xnnpack.fully_connected_nc_qd8_f32_qc4w %input, %xor, transpose_rhs = false, kernel_id = -1 : (tensor<2x4xi8>, tensor<6x4xi4>) -> tensor<2x6xf32>
  return %out : tensor<2x6xf32>
}
