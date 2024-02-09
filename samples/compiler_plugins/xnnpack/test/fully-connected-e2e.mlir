// RUN: iree-compile --iree-hal-target-backends=llvm-cpu --iree-plugin=xnnpack %s | \
// RUN: iree-run-module \
// RUN:     --device=local-sync \
// RUN:     --executable_plugin=$IREE_BINARY_DIR/samples/custom_dispatch/xnnpack/plugin/system_plugin$IREE_DYLIB_EXT \
// RUN:     --module=- \
// RUN:     --function=main \
// RUN:     --input=1x2x8xi8=1 \
// RUN:     --input=4x8xi8=2 --xnnpack_thread_count=2 | \
// RUN: FileCheck %s --check-prefix=CHECK-SYSTEM

// CHECK-SYSTEM: EXEC @main
// CHECK-SYSTEM: 1x2x4xf32={{\[}}[16 16 16 16][16 16 16 16]]
func.func @main(%input : tensor<1x2x8xi8>, %kernel : tensor<4x8xi8>) -> tensor<1x2x4xf32> {
  // TODO: Avoid doing this XOR.
  // Currently, XNNPACK assumes the kernel is an `ui4`
  // tensor. However, doing an `ui8->i4` conversion does not result in
  // a `ui4` tensor, and a `ui8->ui4` conversion is currently not
  // handled by IREE. Therefore, we manually convert to unsign by
  // XORing.
  %c8 = stablehlo.constant dense<8> : tensor<4x8xi8>
  %kernel_adjusted = stablehlo.xor %kernel, %c8 : (tensor<4x8xi8>, tensor<4x8xi8>) -> tensor<4x8xi8>
  %kernel_i4 = stablehlo.convert %kernel_adjusted : (tensor<4x8xi8>) -> tensor<4x8xi4>
  %c = xnnpack.fully_connected_nc_qd8_f32_qc4w %input, %kernel_i4 kernel_needs_transpose = false : (tensor<1x2x8xi8>, tensor<4x8xi4>) -> tensor<1x2x4xf32>
  func.return %c : tensor<1x2x4xf32>
}
