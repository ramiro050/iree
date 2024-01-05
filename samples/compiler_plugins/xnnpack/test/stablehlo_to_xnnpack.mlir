func.func @batch_matrix_multiply(%a : tensor<1x100x200xi8>, %b : tensor<200x300xi8>) -> tensor<1x100x300xi32> {
  %out = stablehlo.dot_general %a, %b, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x100x200xi8>, tensor<200x300xi8>) -> tensor<1x100x300xi32>
  func.return %c : tensor<1x100x300xi32>
}

