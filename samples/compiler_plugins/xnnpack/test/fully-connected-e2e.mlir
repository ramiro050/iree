// RUN: iree-compile --iree-hal-target-backends=llvm-cpu --iree-plugin=xnnpack %s | \
// RUN: iree-run-module \
// RUN:     --device=local-sync \
// RUN:     --executable_plugin=$IREE_BINARY_DIR/samples/custom_dispatch/xnnpack/plugin/system_plugin$IREE_DYLIB_EXT \
// RUN:     --module=- \
// RUN:     --function=main \
// RUN:     --input=1x2x8xi8=1 \
// RUN:     --input=4x8xi8=2 \
// RUN:     --input=4x8xi8=8 | \
// llvm-lit does not like checking for '[['
// RUN:     sed 's/\[/(/g' | tee /tmp/dump.txt | \
// RUN: FileCheck %s --check-prefix=CHECK-SYSTEM

// CHECK-SYSTEM: EXEC @main
// CHECK-SYSTEM: 1x2x4xf32=((16 16 16 16](16 16 16 16]]
func.func @main(%a : tensor<1x2x8xi8>, %b : tensor<4x8xi8>, %shift : tensor<4x8xi8>) -> tensor<1x2x4xf32> {
  // TODO: Avoid doing this shift.
  // Currently, XNNPACK will shift the weight by -8, so here the
  // weight is shifted in the other direction.
  %b_shift = stablehlo.add %b, %shift : (tensor<4x8xi8>, tensor<4x8xi8>) -> tensor<4x8xi8>
  %b_i4 = stablehlo.convert %b_shift : (tensor<4x8xi8>) -> tensor<4x8xi4>
  %c = xnnpack.fully_connected_nc_qd8_f32_qc4w %a, %b_i4 : (tensor<1x2x8xi8>, tensor<4x8xi4>) -> tensor<1x2x4xf32>
  func.return %c : tensor<1x2x4xf32>
}
