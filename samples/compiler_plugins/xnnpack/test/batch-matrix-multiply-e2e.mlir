// RUN: iree-compile --iree-hal-target-backends=llvm-cpu --iree-plugin=xnnpack %s | \
// RUN: iree-run-module \
// RUN:     --device=local-sync \
// RUN:     --executable_plugin=$IREE_BINARY_DIR/samples/custom_dispatch/xnnpack/plugin/system_plugin$IREE_DYLIB_EXT \
// RUN:     --module=- \
// RUN:     --function=main \
// RUN:     --input=2x2x2xf32=2 \
// RUN:     --input=2x2x2xf32=4 \
// RUN:     --xnnpack_thread_count=1 |\
// RUN: FileCheck %s --check-prefix=CHECK-SYSTEM

// CHECK-SYSTEM: EXEC @main
// CHECK-SYSTEM: 2x2x2xf32={{\[}}[16 16][16 16]]{{\[}}[16 16][16 16]]
func.func @main(%a : tensor<?x2x2xf32>, %b : tensor<?x2x2xf32>) -> tensor<?x2x2xf32> {
  %c = xnnpack.batch_matrix_multiply %a, %b : (tensor<?x2x2xf32>, tensor<?x2x2xf32>) -> tensor<?x2x2xf32>
  func.return %c : tensor<?x2x2xf32>
}
