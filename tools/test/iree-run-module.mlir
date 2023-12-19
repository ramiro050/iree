// RUN: (iree-compile --iree-hal-target-backends=vmvx %s | iree-run-module --device=local-task --module=- --function=abs --input="2xf32=-2 3") | FileCheck %s
// RUN: (iree-compile --iree-hal-target-backends=llvm-cpu %s | iree-run-module --device=local-task --module=- --function=abs --input="2xf32=-2 3") | FileCheck %s

// CHECK-LABEL: EXEC @abs
func.func @abs(%input : tensor<2xf32>) -> (tensor<2xf32>) {
  %result = math.absf %input : tensor<2xf32>
  return %result : tensor<2xf32>
}
  // INPUT-BUFFERS: result[1]: hal.buffer_view
  // INPUT-BUFFERS-NEXT: 2xf32=-2.0 3.0
