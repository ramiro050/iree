// This example demonstrates calling dynamically imported functions in the
// runtime through the use of ukernels. This is calling the same function
// as `system_example.mlir`, but using the `iree_codegen.ukernel.generic`.
// This is an example of how ukernels can be called from code generated
// by IREE.

// RUN: iree-compile --iree-hal-target-backends=llvm-cpu %s | \
// RUN: iree-run-module \
// RUN:     --device=local-sync \
// RUN:     --executable_plugin=$IREE_BINARY_DIR/samples/custom_dispatch/xnnpack/plugin/system_plugin$IREE_DYLIB_EXT \
// RUN:     --function="main" \
// RUN:     --module=- \
// RUN:     --input=8xf32=2 \
// RUN:     --input=8xf32=4 \
// RUN:     --xnnpack_thread_count=1 |\
// RUN: FileCheck %s --check-prefix=CHECK-SYSTEM

// CHECK-SYSTEM: EXEC @main
// xnnpack.multiply2_workgroup
// CHECK_SYSTEM: mul2[0](2 * 4 = 8)
// CHECK_SYSTEM: mul2[1](2 * 4 = 8)
// CHECK_SYSTEM: mul2[2](2 * 4 = 8)
// CHECK_SYSTEM: mul2[3](2 * 4 = 8)
// CHECK_SYSTEM: mul2[4](2 * 4 = 8)
// CHECK_SYSTEM: mul2[5](2 * 4 = 8)
// CHECK_SYSTEM: mul2[6](2 * 4 = 8)
// CHECK_SYSTEM: mul2[7](2 * 4 = 8)
// CHECK_SYSTEM: result[0]: hal.buffer_view
// CHECK_SYSTEM: 8xf32=8 8 8 8 8 8 8 8

func.func @main(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) -> tensor<?xf32> {
  %c0 = arith.constant 0 : index
  %dim = tensor.dim %arg0, %c0 : tensor<?xf32>
  %c0_0 = arith.constant 0 : index
  %dim_1 = tensor.dim %arg1, %c0_0 : tensor<?xf32>
  %0 = tensor.empty(%dim) : tensor<?xf32>
  %1 = flow.dispatch.region -> (tensor<?xf32>{%dim}) {
    %2 = iree_codegen.ukernel.generic "xnnpack.multiply2_workgroup" ins(%arg0, %arg1 : tensor<?xf32>, tensor<?xf32>) outs(%0 : tensor<?xf32>) (%dim, %dim_1, %dim : index, index, index) -> tensor<?xf32>
    flow.return %2 : tensor<?xf32>
  } count() -> (index, index, index) {
    %c1 = arith.constant 1 : index
    flow.return %c1, %c1, %c1 : index, index, index
  }
  return %1 : tensor<?xf32>
}
