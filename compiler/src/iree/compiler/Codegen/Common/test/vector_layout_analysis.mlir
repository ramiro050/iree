// RUN: iree-opt -iree-transform-dialect-interpreter --split-input-file %s --verify-diagnostics

#layout = #iree_vector_ext.layout<<[VECTORY], [16]>, <[VECTORX], [16]>>

// Propagate the layout from transfer_read to everyone.
builtin.module attributes { transform.with_named_sequence } {
  func.func @propagate_simple(%arr: memref<16x16xf16>, %a: vector<16x16xf16>, %b: vector<16x16xf16>) -> vector<16x16xf16> {
    %c0 = arith.constant 0 : index
    %cst_0 = arith.constant 0.0 : f16
    %root = vector.transfer_read %arr[%c0, %c0], %cst_0 {in_bounds = [true, true], "__vector_layout_test_anchor_result_0" = #layout} : memref<16x16xf16>, vector<16x16xf16>
    // expected-remark @above {{layout of result #0 is #iree_vector_ext.layout<<[ VECTORY], [16]>, <[ VECTORX], [16]>>}}
    %c = arith.mulf %root, %b : vector<16x16xf16>
    // expected-remark @above {{layout of result #0 is #iree_vector_ext.layout<<[ VECTORY], [16]>, <[ VECTORX], [16]>>}}
    %d = arith.addf %c, %a : vector<16x16xf16>
    // expected-remark @above {{layout of result #0 is #iree_vector_ext.layout<<[ VECTORY], [16]>, <[ VECTORX], [16]>>}}
    func.return %d : vector<16x16xf16>
  }

  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_vector_layout_analysis %top_level_func : !transform.any_op
    transform.yield 
  }
}

// -----

#layout = #iree_vector_ext.layout<<[VECTORY], [16]>, <[VECTORX], [16]>>

// Enforce the layout from the transfer_write to everyone
builtin.module attributes { transform.with_named_sequence } {
  func.func @enforce_simple(%arr: memref<16x16xf16>, %a: vector<16x16xf16>, %b: vector<16x16xf16>) -> vector<16x16xf16> {
    %c0 = arith.constant 0 : index
    %cst_0 = arith.constant 0.0 : f16
    %cst0 = arith.constant dense<0.0> : vector<16x16xf16>
    // expected-remark @above {{layout of result #0 is #iree_vector_ext.layout<<[ VECTORY], [16]>, <[ VECTORX], [16]>>}}
    %c = arith.mulf %cst0, %b : vector<16x16xf16>
    // expected-remark @above {{layout of result #0 is #iree_vector_ext.layout<<[ VECTORY], [16]>, <[ VECTORX], [16]>>}}
    %d = arith.addf %c, %a : vector<16x16xf16>
    // expected-remark @above {{layout of result #0 is #iree_vector_ext.layout<<[ VECTORY], [16]>, <[ VECTORX], [16]>>}}
    vector.transfer_write %d, %arr[%c0, %c0] {in_bounds = [true, true], "__vector_layout_test_anchor_operand_0" = #layout} : vector<16x16xf16>, memref<16x16xf16>
    func.return %d : vector<16x16xf16>
  }

  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_vector_layout_analysis %top_level_func : !transform.any_op
    transform.yield 
  }
}

// -----

#layout = #iree_vector_ext.layout<<[VECTORY], [16]>, <[VECTORX], [16]>>

// First propagate the layout, and then enforce it up.
builtin.module attributes { transform.with_named_sequence } {
  func.func @propagate_and_enforce(%arr: memref<16x16xf16>, %arr2: memref<16x16xf16>, %a: vector<16x16xf16>, %b: vector<16x16xf16>) -> vector<16x16xf16> {
    %c0 = arith.constant 0 : index
    %cst_0 = arith.constant 0.0 : f16
    %root = vector.transfer_read %arr[%c0, %c0], %cst_0 {in_bounds = [true, true], "__vector_layout_test_anchor_result_0" = #layout} : memref<16x16xf16>, vector<16x16xf16>
    // expected-remark @above {{layout of result #0 is #iree_vector_ext.layout<<[ VECTORY], [16]>, <[ VECTORX], [16]>>}}
    %root2 = vector.transfer_read %arr2[%c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<16x16xf16>, vector<16x16xf16>
    // expected-remark @above {{layout of result #0 is #iree_vector_ext.layout<<[ VECTORY], [16]>, <[ VECTORX], [16]>>}}
    %c = arith.mulf %root, %b : vector<16x16xf16>
    // expected-remark @above {{layout of result #0 is #iree_vector_ext.layout<<[ VECTORY], [16]>, <[ VECTORX], [16]>>}}
    %d = arith.addf %c, %a : vector<16x16xf16>
    // expected-remark @above {{layout of result #0 is #iree_vector_ext.layout<<[ VECTORY], [16]>, <[ VECTORX], [16]>>}}
    %e = arith.divf %d, %root2 : vector<16x16xf16>
    // expected-remark @above {{layout of result #0 is #iree_vector_ext.layout<<[ VECTORY], [16]>, <[ VECTORX], [16]>>}}
    func.return %e : vector<16x16xf16>
  }

  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_vector_layout_analysis %top_level_func : !transform.any_op
    transform.yield 
  }
}

// -----

#layout = #iree_vector_ext.layout<<[VECTORY], [16]>, <[BATCHY, VECTORX], [2, 8]>>

// Propagate and enforce through reduction.
builtin.module attributes { transform.with_named_sequence } {
  func.func @reduction(%arr: memref<16x16xf16>, %arr2: memref<16xf16>, %a: vector<16xf16>, %b: vector<16xf16>) -> vector<16xf16> {
    %c0 = arith.constant 0 : index
    %cst_0 = arith.constant 0.0 : f16
    %cst0_1 = arith.constant dense<0.0> : vector<16xf16>
    // expected-remark @above {{layout of result #0 is #iree_vector_ext.layout<<[ BATCHY,  VECTORX], [2, 8]>>}}
    %root = vector.transfer_read %arr[%c0, %c0], %cst_0 {in_bounds = [true, true], "__vector_layout_test_anchor_result_0" = #layout} : memref<16x16xf16>, vector<16x16xf16>
    // expected-remark @above {{layout of result #0 is #iree_vector_ext.layout<<[ VECTORY], [16]>, <[ BATCHY,  VECTORX], [2, 8]>>}}
    %root2 = vector.transfer_read %arr2[%c0], %cst_0 {in_bounds = [true]} : memref<16xf16>, vector<16xf16>
    // expected-remark @above {{layout of result #0 is #iree_vector_ext.layout<<[ BATCHY,  VECTORX], [2, 8]>>}}
    %root_red = vector.multi_reduction<add>, %root, %cst0_1 [0]  : vector<16x16xf16> to vector<16xf16>
    // expected-remark @above {{layout of result #0 is #iree_vector_ext.layout<<[ BATCHY,  VECTORX], [2, 8]>>}}
    %c = arith.mulf %root_red, %b : vector<16xf16>
    // expected-remark @above {{layout of result #0 is #iree_vector_ext.layout<<[ BATCHY,  VECTORX], [2, 8]>>}}
    %d = arith.addf %c, %a : vector<16xf16>
    // expected-remark @above {{layout of result #0 is #iree_vector_ext.layout<<[ BATCHY,  VECTORX], [2, 8]>>}}
    %e = arith.divf %d, %root2 : vector<16xf16>
    // expected-remark @above {{layout of result #0 is #iree_vector_ext.layout<<[ BATCHY,  VECTORX], [2, 8]>>}}
    func.return %e : vector<16xf16>
  }

  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_vector_layout_analysis %top_level_func : !transform.any_op
    transform.yield 
  }
}

// -----

#layout = #iree_vector_ext.layout<<[VECTORY], [16]>, <[BATCHY, VECTORX], [2, 8]>>

// Propagate and enforce through transpose and then reduction.
builtin.module attributes { transform.with_named_sequence } {
  func.func @transpose_and_reduction(%arr: memref<16x16xf16>, %arr2: memref<16xf16>, %a: vector<16xf16>, %b: vector<16xf16>) -> vector<16xf16> {
    %c0 = arith.constant 0 : index
    %cst_0 = arith.constant 0.0 : f16
    %cst0_1 = arith.constant dense<0.0> : vector<16xf16>
    // expected-remark @above {{layout of result #0 is #iree_vector_ext.layout<<[ VECTORY], [16]>>}}
    %root = vector.transfer_read %arr[%c0, %c0], %cst_0 {in_bounds = [true, true], "__vector_layout_test_anchor_result_0" = #layout} : memref<16x16xf16>, vector<16x16xf16>
    // expected-remark @above {{layout of result #0 is #iree_vector_ext.layout<<[ VECTORY], [16]>, <[ BATCHY,  VECTORX], [2, 8]>>}}
    %root2 = vector.transfer_read %arr2[%c0], %cst_0 {in_bounds = [true]} : memref<16xf16>, vector<16xf16>
    // expected-remark @above {{layout of result #0 is #iree_vector_ext.layout<<[ VECTORY], [16]>>}}
    %root_transpose = vector.transpose %root, [1, 0] : vector<16x16xf16> to vector<16x16xf16>
    // expected-remark @above {{layout of result #0 is #iree_vector_ext.layout<<[ BATCHY,  VECTORX], [2, 8]>, <[ VECTORY], [16]>>}}
    %root_red = vector.multi_reduction<add>, %root_transpose, %cst0_1 [0]  : vector<16x16xf16> to vector<16xf16>
    // expected-remark @above {{layout of result #0 is #iree_vector_ext.layout<<[ VECTORY], [16]>>}}
    %c = arith.mulf %root_red, %b : vector<16xf16>
    // expected-remark @above {{layout of result #0 is #iree_vector_ext.layout<<[ VECTORY], [16]>>}}
    %d = arith.addf %c, %a : vector<16xf16>
    // expected-remark @above {{layout of result #0 is #iree_vector_ext.layout<<[ VECTORY], [16]>>}}
    %e = arith.divf %d, %root2 : vector<16xf16>
    // expected-remark @above {{layout of result #0 is #iree_vector_ext.layout<<[ VECTORY], [16]>>}}
    func.return %e : vector<16xf16>
  }

  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_vector_layout_analysis %top_level_func : !transform.any_op
    transform.yield 
  }
}

// -----

#layoutA = #iree_vector_ext.layout<<[VECTORX], [32]>, <[VECTORY], [64]>>
#layoutB = #iree_vector_ext.layout<<[VECTORX], [128]>, <[VECTORY], [64]>>
#layoutC = #iree_vector_ext.layout<<[VECTORY], [128]>, <[VECTORX], [32]>>

#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d1, d0)>

// Propagate through vector.contract.
builtin.module attributes { transform.with_named_sequence } {
  func.func @contract(%A : vector<32x64xf16>, %B : vector<128x64xf16>, %C : vector<128x32xf32>) -> vector<128x32xf32> {
    // Check if the layout of %C was properly propagated to %D.
    // expected-remark @below {{layout of result #0 is #iree_vector_ext.layout<<[ VECTORY], [128]>, <[ VECTORX], [32]>>}}
    %D = vector.contract
        {indexing_maps = [#map1, #map2, #map3],
         iterator_types = ["parallel", "parallel", "reduction"],
         kind = #vector.kind<add>,
         "__vector_layout_test_anchor_operand_0" = #layoutB,
         "__vector_layout_test_anchor_operand_1" = #layoutA,
         "__vector_layout_test_anchor_operand_2" = #layoutC
        } %B, %A, %C : vector<128x64xf16>, vector<32x64xf16> into vector<128x32xf32>
    func.return %D : vector<128x32xf32>
  }

  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_vector_layout_analysis %top_level_func : !transform.any_op
    transform.yield
  }
}

// -----

#layout = #iree_vector_ext.layout<<[VECTORY], [16]>, <[BATCHY, VECTORX], [2, 8]>>

// Propagate and enforce through scf.for
builtin.module attributes { transform.with_named_sequence } {
  func.func @scffor(%arr: memref<16x16xf16>, %arr2: memref<16xf16>, %a: vector<16xf16>, %b: vector<16xf16>) -> vector<16xf16> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c1024 = arith.constant 1024 : index
    %cst_0 = arith.constant 0.0 : f16
    %cst0_1 = arith.constant dense<0.0> : vector<16xf16>
    // expected-remark @above {{layout of result #0 is #iree_vector_ext.layout<<[ VECTORY], [16]>>}}

    %out = scf.for %iv = %c0 to %c1024 step %c1 iter_args(%arg1 = %cst0_1) -> (vector<16xf16>) {
      // expected-remark @above {{layout of result #0 is #iree_vector_ext.layout<<[ VECTORY], [16]>>}}
      %root = vector.transfer_read %arr[%c0, %c0], %cst_0 {in_bounds = [true, true], "__vector_layout_test_anchor_result_0" = #layout} : memref<16x16xf16>, vector<16x16xf16>
      // expected-remark @above {{layout of result #0 is #iree_vector_ext.layout<<[ VECTORY], [16]>, <[ BATCHY,  VECTORX], [2, 8]>>}}
      %root2 = vector.transfer_read %arr2[%c0], %cst_0 {in_bounds = [true]} : memref<16xf16>, vector<16xf16>
      // expected-remark @above {{layout of result #0 is #iree_vector_ext.layout<<[ VECTORY], [16]>>}}
      %root_transpose = vector.transpose %root, [1, 0] : vector<16x16xf16> to vector<16x16xf16>
      // expected-remark @above {{layout of result #0 is #iree_vector_ext.layout<<[ BATCHY,  VECTORX], [2, 8]>, <[ VECTORY], [16]>>}}
      %root_red = vector.multi_reduction<add>, %root_transpose, %arg1 [0]  : vector<16x16xf16> to vector<16xf16>
      // expected-remark @above {{layout of result #0 is #iree_vector_ext.layout<<[ VECTORY], [16]>>}}
      %c = arith.mulf %root_red, %b : vector<16xf16>
      // expected-remark @above {{layout of result #0 is #iree_vector_ext.layout<<[ VECTORY], [16]>>}}
      %d = arith.addf %c, %a : vector<16xf16>
      // expected-remark @above {{layout of result #0 is #iree_vector_ext.layout<<[ VECTORY], [16]>>}}
      %e = arith.divf %d, %root2 : vector<16xf16>
      // expected-remark @above {{layout of result #0 is #iree_vector_ext.layout<<[ VECTORY], [16]>>}}
      scf.yield %e : vector<16xf16>
    }

    func.return %out : vector<16xf16>
  }

  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_vector_layout_analysis %top_level_func : !transform.any_op
    transform.yield
  }
}
