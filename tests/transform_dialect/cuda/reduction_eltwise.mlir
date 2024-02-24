!in_tensor_t = tensor<8x64xf32>
!out_tensor_t = tensor<8xf32>

func.func @reduce(%arg : !in_tensor_t) -> (!out_tensor_t) {
  %cst = arith.constant -0.000000e+00 : f32

  %0 = tensor.empty() : !out_tensor_t
  %1 = linalg.fill ins(%cst : f32) outs(%0 : !out_tensor_t) -> !out_tensor_t
  %5 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0)>],
    iterator_types = ["parallel", "reduction"]}
    ins(%arg : !in_tensor_t) outs(%1 : !out_tensor_t) {
      ^bb0(%arg3: f32, %arg4: f32):
        %4 = arith.addf %arg3, %arg4 : f32
        linalg.yield %4 : f32
      } -> !out_tensor_t

  %6 = tensor.empty() : !out_tensor_t
  %7 = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>,
                     affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]}
    ins(%5 : !out_tensor_t) outs(%6 : !out_tensor_t) {
    ^bb0(%arg3: f32, %arg4: f32):
      %4 = math.sqrt %arg3 : f32
      linalg.yield %4 : f32
    } -> !out_tensor_t
  return %7 : !out_tensor_t
}

// RUN: iree-compile %s --iree-hal-target-backends=cuda \
// RUN:     --iree-codegen-llvmgpu-enable-transform-dialect-jit=false \
// RUN:     --iree-codegen-transform-dialect-library=%p/reduction_eltwise_codegen_spec.mlir@codegen | \
// RUN: iree-run-module --module=- --function=reduce --device=cuda --input="8x64xf32=1" |\
// RUN: FileCheck %s --check-prefix=EXEC

/// Note: the current --iree-codegen-llvmgpu-enable-transform-dialect-jit
/// only works for exactly this reduction atm.
// RUN: iree-compile %s --iree-hal-target-backends=cuda | \
// RUN: iree-run-module --module=- --function=reduce --device=cuda --input="8x64xf32=1" |\
// RUN: FileCheck %s --check-prefix=EXEC

//      EXEC: result[0]: hal.buffer_view
// EXEC-NEXT: 8xf32=8 8 8 8 8 8 8 8
