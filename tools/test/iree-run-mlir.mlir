// RUN: (iree-run-mlir --Xcompiler,iree-hal-target-backends=vmvx %s --input=f32=-2) | FileCheck %s
// RUN: iree-run-mlir --Xcompiler,iree-hal-target-backends=llvm-cpu %s --input=f32=-2 | FileCheck %s

// CHECK-LABEL: EXEC @abs
func.func @abs(%input : tensor<f32>) -> (tensor<f32>) {
  %result = math.absf %input : tensor<f32>
  return %result : tensor<f32>
}
// CHECK: f32=2
