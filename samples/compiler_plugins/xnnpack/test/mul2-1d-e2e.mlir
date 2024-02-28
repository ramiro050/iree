// RUN: iree-compile --iree-hal-target-backends=llvm-cpu --iree-plugin=xnnpack --compile-to=flow %s | \
// RUN: iree-opt --inline | \
// RUN: iree-compile --iree-hal-target-backends=llvm-cpu --compile-from=flow - | \
// RUN: iree-run-module \
// RUN:     --device=local-sync \
// RUN:     --executable_plugin=$IREE_BINARY_DIR/samples/custom_dispatch/xnnpack/plugin/system_plugin$IREE_DYLIB_EXT \
// RUN:     --module=- \
// RUN:     --function=main \
// RUN:     --input=8xf32=2 \
// RUN:     --input=8xf32=4 \
// RUN:     --xnnpack_thread_count=1 |\
// RUN: FileCheck %s --check-prefix=CHECK-SYSTEM

// CHECK-SYSTEM: EXEC @main
// CHECK-SYSTEM: mul2[0](2 * 4 = 8)
// CHECK-SYSTEM: mul2[1](2 * 4 = 8)
// CHECK-SYSTEM: mul2[2](2 * 4 = 8)
// CHECK-SYSTEM: mul2[3](2 * 4 = 8)
// CHECK-SYSTEM: mul2[4](2 * 4 = 8)
// CHECK-SYSTEM: mul2[5](2 * 4 = 8)
// CHECK-SYSTEM: mul2[6](2 * 4 = 8)
// CHECK-SYSTEM: mul2[7](2 * 4 = 8)
// CHECK-SYSTEM: 8xf32=8 8 8 8 8 8 8 8
func.func @main(%a : tensor<8xf32>, %b : tensor<8xf32>) -> tensor<8xf32> {
  %c = xnnpack.multiply2 %a, %b : (tensor<8xf32>, tensor<8xf32>) -> tensor<8xf32>
  func.return %c : tensor<8xf32>
}
