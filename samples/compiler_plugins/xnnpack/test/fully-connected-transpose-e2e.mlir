// RUN: iree-compile --iree-hal-target-backends=llvm-cpu --iree-plugin=xnnpack --compile-to=flow %s | \
// RUN: iree-opt --inline | \
// RUN: iree-compile --iree-hal-target-backends=llvm-cpu --compile-from=flow - | \
// RUN: iree-run-module \
// RUN:     --device=local-sync \
// RUN:     --executable_plugin=$IREE_BINARY_DIR/samples/custom_dispatch/xnnpack/plugin/system_plugin$IREE_DYLIB_EXT \
// RUN:     --module=- \
// RUN:     --function=main \
// RUN:     --input=2x2xi8=1 --xnnpack_thread_count=2 | \
// RUN: FileCheck %s --check-prefix=CHECK-SYSTEM

// CHECK-SYSTEM: EXEC @main
// CHECK-SYSTEM: 2x4xf32=[-4 -2 0 2][-4 -2 0 2]
func.func @main(%input : tensor<2x2xi8>) -> tensor<2x4xf32> {
  // XNNPACK expects the `kernel` tensor to have unsigned values with
  // zero point of 8. The XOR operation transforms the `kernel` tensor
  // from having zero point of 0 to having zero point of 8 (i.e. going
  // from signed i4 space to unsigned i4 space).
  %kernel = stablehlo.constant dense<[[-4, -3, -2, -1], [0, 1, 2, 3]]> : tensor<2x4xi4>
  %c8 = stablehlo.constant dense<8> : tensor<2x4xi8>
  %c8_int4 = stablehlo.convert %c8 : (tensor<2x4xi8>) -> tensor<2x4xi4>
  %kernel_adjusted = stablehlo.xor %kernel, %c8_int4 : (tensor<2x4xi4>, tensor<2x4xi4>) -> tensor<2x4xi4>
  %c = xnnpack.fully_connected_nc_qd8_f32_qc4w %input, %kernel_adjusted, transpose_rhs = true, kernel_id = -1 : (tensor<2x2xi8>, tensor<2x4xi4>) -> tensor<2x4xf32>
  func.return %c : tensor<2x4xf32>
}
