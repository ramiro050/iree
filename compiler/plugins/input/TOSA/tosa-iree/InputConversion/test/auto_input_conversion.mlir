// RUN: iree-compile --compile-to=input --split-input-file %s | FileCheck %s

// Check that the auto input conversion pipeline uses this plugin.

// CHECK-LABEL: func.func @simple_add_tosa
// CHECK:  arith.addi
func.func @simple_add_tosa(%arg0: tensor<2x2xi32>, %arg1: tensor<2x2xi32>) -> tensor<2x2xi32> {
  %0 = tosa.add %arg0, %arg1 : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
  return %0 : tensor<2x2xi32>
}
