// RUN: iree-compile --iree-hal-target-backends=llvm-cpu --iree-plugin=xnnpack %s | \
// RUN: iree-run-module \
// RUN:     --device=local-sync \
// RUN:     --executable_plugin=$IREE_BINARY_DIR/samples/custom_dispatch/xnnpack/plugin/system_plugin$IREE_DYLIB_EXT \
// RUN:     --module=- \
// RUN:     --function=main \
// RUN:     --input=1x2x8xi8=1 \
// RUN:     --input=8x4xi8=2 --xnnpack_thread_count=2 | \
// RUN: FileCheck %s --check-prefix=CHECK-SYSTEM

// CHECK-SYSTEM: EXEC @main
// CHECK-SYSTEM: 1x2x4xf32={{\[}}[16 16 16 16][16 16 16 16]]
func.func @main(%input : tensor<1x2x8xi8>, %kernel : tensor<8x4xi8>) -> tensor<1x2x4xf32> {
  // XNNPACK expects the `kernel` tensor to have unsigned values with
  // zero point of 8. The XOR operation transforms the `kernel` tensor
  // from having zero point of 0 to having zero point of 8 (i.e. going
  // from signed i4 space to unsigned i4 space).
  %c8 = stablehlo.constant dense<8> : tensor<8x4xi8>
  %kernel_adjusted = stablehlo.xor %kernel, %c8 : (tensor<8x4xi8>, tensor<8x4xi8>) -> tensor<8x4xi8>
  %kernel_i4 = stablehlo.convert %kernel_adjusted : (tensor<8x4xi8>) -> tensor<8x4xi4>
  %c = xnnpack.fully_connected_nc_qd8_f32_qc4w %input, %kernel_i4 transpose_rhs = true : (tensor<1x2x8xi8>, tensor<8x4xi4>) -> tensor<1x2x4xf32>
  func.return %c : tensor<1x2x4xf32>
}
