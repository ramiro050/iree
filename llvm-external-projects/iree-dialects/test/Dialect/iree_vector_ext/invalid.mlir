// RUN: iree-dialects-opt --split-input-file --verify-diagnostics %s

#row_layout1 = #iree_vector_ext.per_dim_layout<[BATCHX, LANEX, VECTORY], [1, 1, 1]>
#col_layout1 = #iree_vector_ext.per_dim_layout<[BATCHY, LANEY, VECTORX], [4, 2, 4]>
#layout1 = #iree_vector_ext.layout<#row_layout1, #col_layout1>
#layout2 = #iree_vector_ext.layout<#col_layout1, #col_layout1>
func.func @invalid_desired_layout(%lhs: memref<32x32xf16>, %rhs: memref<32x32xf16>) -> vector<32x32xf16> {
  %cst_0 = arith.constant 0.0 : f16
  %c0 = arith.constant 0 : index
  %result = vector.transfer_read %lhs[%c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<32x32xf16>, vector<32x32xf16>
  // expected-error @+1 {{The desired layout shape cannot be distributed over the given vector shape}}
  %2 = iree_vector_ext.layout_conflict_resolution %result {desiredLayout = #layout1, sourceLayout = #layout2} : vector<32x32xf16> -> vector<32x32xf16>
  return %2 : vector<32x32xf16>
}

// -----

#row_layout1 = #iree_vector_ext.per_dim_layout<[BATCHX, LANEX, VECTORY], [1, 1, 1]>
#col_layout1 = #iree_vector_ext.per_dim_layout<[BATCHY, LANEY, VECTORX], [4, 2, 4]>
#layout1 = #iree_vector_ext.layout<#row_layout1, #col_layout1>
#layout2 = #iree_vector_ext.layout<#col_layout1, #col_layout1>
func.func @invalid_source_layout(%lhs: memref<32x32xf16>, %rhs: memref<32x32xf16>) -> vector<32x32xf16> {
  %cst_0 = arith.constant 0.0 : f16
  %c0 = arith.constant 0 : index
  %result = vector.transfer_read %lhs[%c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<32x32xf16>, vector<32x32xf16>
  // expected-error @+1 {{The source layout shape cannot be distributed over the given vector shape}}
  %2 = iree_vector_ext.layout_conflict_resolution %result {desiredLayout = #layout2, sourceLayout = #layout1} : vector<32x32xf16> -> vector<32x32xf16>
  return %2 : vector<32x32xf16>
}

// -----

func.func @invalid_to_simd_vector_element_type(%simd : vector<2x2xf16>) -> vector<64xf32> {
  // expected-error @+1 {{requires the same element type for all operands and results}}
  %simt = iree_vector_ext.to_simd %simd : vector<2x2xf16> -> vector<64xf32>
  func.return %simt : vector<64xf32>
}

// -----

func.func @invalid_to_simt_vector_element_type(%simt : vector<64xf32>) -> vector<2x2xf16> {
  // expected-error @+1 {{requires the same element type for all operands and results}}
  %simd = iree_vector_ext.to_simt %simt : vector<64xf32> -> vector<2x2xf16>
  func.return %simd : vector<2x2xf16>
}
