// RUN: iree-opt --iree-plugin=xnnpack --verify-diagnostics --split-input-file %s

func.func @batch_matrix_multiply$small_rank(%a : tensor<?x?x?xf32>, %b : tensor<?x?xf32>) -> tensor<?x?x?xf32> {
  // expected-error@+1 {{expected operands to have rank >= 3}}
  %c = xnnpack.batch_matrix_multiply %a, %b : (tensor<?x?x?xf32>, tensor<?x?xf32>) -> tensor<?x?x?xf32>
  func.return %c : tensor<?x?x?xf32>
}

// -----

func.func @batch_matrix_multiply$unequal_rank(%a : tensor<?x?x?xf32>, %b : tensor<?x?x?x?xf32>) -> tensor<?x?x?xf32> {
  // expected-error@+1 {{expected operands to have the same rank}}
  %c = xnnpack.batch_matrix_multiply %a, %b : (tensor<?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?xf32>
  func.return %c : tensor<?x?x?xf32>
}

// -----

func.func @batch_matrix_multiply$unequal_batch_dims(%a : tensor<3x?x?xf32>, %b : tensor<2x?x?xf32>) -> tensor<?x?x?xf32> {
  // expected-error@+1 {{expected first N-2 dimensions to match}}
  %c = xnnpack.batch_matrix_multiply %a, %b : (tensor<3x?x?xf32>, tensor<2x?x?xf32>) -> tensor<?x?x?xf32>
  func.return %c : tensor<?x?x?xf32>
}

// -----

func.func @batch_matrix_multiply$unequal_reduction_dim(%a : tensor<?x?x2xf32>, %b : tensor<?x3x?xf32>) -> tensor<?x?x?xf32> {
  // expected-error@+1 {{expected reduction dimension to be the same}}
  %c = xnnpack.batch_matrix_multiply %a, %b : (tensor<?x?x2xf32>, tensor<?x3x?xf32>) -> tensor<?x?x?xf32>
  func.return %c : tensor<?x?x?xf32>
}
