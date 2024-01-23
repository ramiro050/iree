// RUN: iree-compile --iree-hal-target-backends=llvm-cpu --iree-plugin=xnnpack %s --xnnpack-threads=2 | \
// RUN: iree-run-module \
// RUN:     --device=local-sync \
// RUN:     --executable_plugin=$IREE_BINARY_DIR/samples/custom_dispatch/xnnpack/plugin/system_plugin$IREE_DYLIB_EXT \
// RUN:     --module=- \
// RUN:     --function=main \
// RUN:     --input=1x2x8xi8=1 \
// RUN:     --input=4x8xi8=2 \
// RUN:     --input=4x8xi8=8 | \
// RUN: FileCheck %s --check-prefix=CHECK-SYSTEM

// CHECK-SYSTEM: EXEC @main
// CHECK-SYSTEM: 1x2x4xf32={{\[}}[16 16 16 16][16 16 16 16]]
func.func @main(%input : tensor<1x2x8xi8>, %kernel : tensor<4x8xi8>, %offset : tensor<4x8xi8>) -> tensor<1x2x4xf32> {
  // TODO: Avoid doing this addition.
  // Currently, XNNPACK will subtract the kernel by 8, so here 8 is added to the weight.
  %kernel_adjusted = stablehlo.add %kernel, %offset : (tensor<4x8xi8>, tensor<4x8xi8>) -> tensor<4x8xi8>
  %kernel_i4 = stablehlo.convert %kernel_adjusted : (tensor<4x8xi8>) -> tensor<4x8xi4>
  %c = xnnpack.fully_connected_nc_qd8_f32_qc4w %input, %kernel_i4 : (tensor<1x2x8xi8>, tensor<4x8xi4>) -> tensor<1x2x4xf32>
  func.return %c : tensor<1x2x4xf32>
}
