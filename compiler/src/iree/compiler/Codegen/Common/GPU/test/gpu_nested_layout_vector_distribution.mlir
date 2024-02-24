// RUN: iree-opt --iree-transform-dialect-interpreter --split-input-file --canonicalize --cse %s | FileCheck %s

#layout_row_major = #iree_vector_ext.nested_layout<
  subgroups_per_workgroup = [1, 1],
  batches_per_subgroup    = [2, 2],
  outers_per_batch        = [1, 1],
  threads_per_outer       = [8, 1],
  elements_per_thread     = [1, 8],

  batch_order             = [1, 0],

  subgroup_basis          = [1, 1],
  thread_basis            = [8, 1]
>

// CHECK: #[[$MAP:.+]] = affine_map<()[s0] -> (s0 + 8)>
// CHECK-LABEL: @distribute_transfer_read_row_major
func.func @distribute_transfer_read_row_major(%arg0: memref<4x4xf16>) -> vector<16x16xf16> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f16
  %root = vector.transfer_read %arg0[%c0, %c0], %cst
          {in_bounds = [false, false],
           "__vector_layout_test_anchor_result_0" = #layout_row_major}
                  : memref<4x4xf16>, vector<16x16xf16>
  func.return %root : vector<16x16xf16>
}

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}

// CHECK: %[[ACC:.+]] = arith.constant dense<0.000000e+00> : vector<2x2x1x1x1x8xf16>
// CHECK: %[[IDX:.+]] = gpu.thread_id  x
// CHECK: %[[IDS:.+]]:4 = affine.delinearize_index %[[IDX]] into (%c1, %c1, %c8, %c1) : index, index, index, index
// CHECK: vector.transfer_read %arg0[%[[IDS]]#2, %c0], {{.*}} : memref<4x4xf16>, vector<1x8xf16>
// CHECK: vector.insert_strided_slice %{{.*}}, %[[ACC]] {offsets = [0, 0, 0, 0, 0, 0], strides = [1, 1]} : vector<1x8xf16> into vector<2x2x1x1x1x8xf16>
// CHECK: vector.transfer_read %arg0[%[[IDS]]#2, %c8]
// CHECK: vector.insert_strided_slice {{.*}} {offsets = [1, 0, 0, 0, 0, 0]
// CHECK: %[[ID_PLUS_BATCH1:.+]] = affine.apply #[[$MAP]]()[%[[IDS]]#2]
// CHECK: vector.transfer_read %arg0[%[[ID_PLUS_BATCH1]], %c0]
// CHECK: vector.insert_strided_slice {{.*}} {offsets = [0, 1, 0, 0, 0, 0]
// CHECK: vector.transfer_read %arg0[%[[ID_PLUS_BATCH1]], %c8]
// CHECK: vector.insert_strided_slice {{.*}} {offsets = [1, 1, 0, 0, 0, 0]
// CHECK: iree_vector_ext.to_simd %{{.*}} : vector<2x2x1x1x1x8xf16> -> vector<16x16xf16>

// -----

#layout_col_major = #iree_vector_ext.nested_layout<
  subgroups_per_workgroup = [1, 1],
  batches_per_subgroup    = [1, 2],
  outers_per_batch        = [1, 1],
  threads_per_outer       = [4, 8],
  elements_per_thread     = [4, 1],

  batch_order             = [1, 0],
  element_order           = [1, 0],

  subgroup_basis          = [1, 1],
  thread_basis            = [4, 8]
>

// CHECK-DAG: #[[$MAP:.+]] = affine_map<()[s0] -> (s0 * 4)>
// CHECK-DAG: #[[$MAP1:.+]] = affine_map<()[s0] -> (s0 + 8)>

// CHECK-LABEL: @distribute_transfer_read_col_major
func.func @distribute_transfer_read_col_major(%arg0: memref<32x32xf16>) -> vector<16x16xf16> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f16
  %root = vector.transfer_read %arg0[%c0, %c0], %cst
          {in_bounds = [true, true],
           "__vector_layout_test_anchor_result_0" = #layout_col_major}
                  : memref<32x32xf16>, vector<16x16xf16>
  func.return %root : vector<16x16xf16>
}

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}

// CHECK: %[[IDS:.+]]:4 = affine.delinearize_index %{{.*}} into (%c1, %c1, %c4, %c8) : index, index, index, index
// CHECK: %[[LANEY:.+]] = affine.apply #[[$MAP]]()[%[[IDS]]#2]
// CHECK: %[[RD00:.+]] = vector.transfer_read %arg0[%[[LANEY:.+]], %[[IDS]]#3], {{.*}} : memref<32x32xf16>, vector<4x1xf16>
// CHECK: %[[ELEM_ORDER:.+]] = vector.transpose %[[RD00]], [1, 0] : vector<4x1xf16> to vector<1x4xf16>
// CHECK: vector.insert_strided_slice %[[ELEM_ORDER]], %{{.*}} {offsets = [0, 0, 0, 0, 0, 0], strides = [1, 1]} : vector<1x4xf16> into vector<2x1x1x1x1x4xf16>
// CHECK: %[[LANEX_PLUS_BATCH:.+]] = affine.apply #[[$MAP1]]()[%[[IDS]]#3]
// CHECK: vector.transfer_read %arg0[%[[LANEY]], %[[LANEX_PLUS_BATCH]]], %{{.*}} {in_bounds = [true, true]} : memref<32x32xf16>, vector<4x1xf16>
// CHECK: vector.transpose %{{.*}}, [1, 0] : vector<4x1xf16> to vector<1x4xf16>
// CHECK: vector.insert_strided_slice {{.*}} {offsets = [1, 0, 0, 0, 0, 0]
// CHECK: iree_vector_ext.to_simd %{{.*}} : vector<2x1x1x1x1x4xf16> -> vector<16x16xf16>

// -----

#layout_row_major = #iree_vector_ext.nested_layout<
  subgroups_per_workgroup = [1, 1],
  batches_per_subgroup    = [2, 2],
  outers_per_batch        = [1, 1],
  threads_per_outer       = [8, 1],
  elements_per_thread     = [1, 8],

  batch_order             = [1, 0],

  subgroup_basis          = [1, 1],
  thread_basis            = [8, 1]
>

// CHECK-DAG: #[[$MAP:.+]] = affine_map<()[s0, s1] -> (s0 + s1)>
// CHECK-DAG: #[[$MAP1:.+]] = affine_map<()[s0] -> (s0 + 8)>
// CHECK-DAG: #[[$MAP2:.+]] = affine_map<()[s0, s1] -> (s0 + s1 + 8)>

func.func @distribute_transfer_read_row_major_with_nontrivial_index(%a: index, %b: index, %arg0: memref<32x32x32x32xf16>) -> vector<16x16xf16> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f16
  %root = vector.transfer_read %arg0[%c0, %c0, %a, %b], %cst
          {in_bounds = [true, true],
           permutation_map = affine_map<(d0, d1, d2, d3) -> (d2, d3)>,
           "__vector_layout_test_anchor_result_0" = #layout_row_major}
                  : memref<32x32x32x32xf16>, vector<16x16xf16>
  func.return %root : vector<16x16xf16>
}

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}
// CHECK-LABEL: @distribute_transfer_read_row_major_with_nontrivial_index
// CHECK-SAME:    %[[I0:.+]]: index, %[[I1:.+]]: index

// CHECK: %[[IDS:.+]]:4 = affine.delinearize_index %{{.*}} into (%c1, %c1, %c8, %c1) : index, index, index, index
// CHECK: %[[OFF0:.+]] = affine.apply #[[$MAP]]()[%[[IDS]]#2, %[[I0]]]
// CHECK: vector.transfer_read %{{.*}}[%c0, %c0, %[[OFF0]], %[[I1]]]
// CHECK: %[[OFF1:.+]] = affine.apply #[[$MAP1]]()[%[[I1]]]
// CHECK: vector.transfer_read %{{.*}}[%c0, %c0, %[[OFF0]], %[[OFF1]]]
// CHECK: %[[OFF2:.+]] = affine.apply #[[$MAP2]]()[%[[IDS]]#2, %[[I0]]]
// CHECK: vector.transfer_read %{{.*}}[%c0, %c0, %[[OFF2]], %[[I1]]]
// CHECK: vector.transfer_read %{{.*}}[%c0, %c0, %[[OFF2]], %[[OFF1]]]

// -----

#layout_col_major = #iree_vector_ext.nested_layout<
  subgroups_per_workgroup = [1, 1],
  batches_per_subgroup    = [1, 2],
  outers_per_batch        = [1, 1],
  threads_per_outer       = [4, 8],
  elements_per_thread     = [4, 1],

  batch_order             = [1, 0],
  element_order           = [1, 0],

  subgroup_basis          = [1, 1],
  thread_basis            = [4, 8]
>

func.func @distribute_transfer_read_col_major_with_broadcast(%a: index, %b: index, %arg0: memref<32x32x32x32xf16>) -> vector<16x16xf16> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f16
  %root = vector.transfer_read %arg0[%c0, %c0, %a, %b], %cst
          {in_bounds = [true, true],
           permutation_map = affine_map<(d0, d1, d2, d3) -> (0, 0)>,
           "__vector_layout_test_anchor_result_0" = #layout_col_major}
                  : memref<32x32x32x32xf16>, vector<16x16xf16>
  func.return %root : vector<16x16xf16>
}

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}

// CHECK: #[[$MAP:.+]] = affine_map<(d0, d1, d2, d3) -> (0, 0)>

// CHECK-LABEL: @distribute_transfer_read_col_major_with_broadcast
// CHECK-SAME:    %[[I0:.+]]: index, %[[I1:.+]]: index

// CHECK: %[[BROADCAST_READ:.+]] = vector.transfer_read %{{.*}}[%c0, %c0, %[[I0]], %[[I1]]], %{{.*}} permutation_map = #[[$MAP]]
// CHECK: %[[UNIT:.+]] = vector.transpose %[[BROADCAST_READ]], [1, 0] : vector<4x1xf16> to vector<1x4xf16>
// CHECK: vector.insert_strided_slice %[[UNIT]], %{{.*}} {offsets = [0, 0, 0, 0, 0, 0]
// CHECK: vector.insert_strided_slice %[[UNIT]], %{{.*}} {offsets = [1, 0, 0, 0, 0, 0]

// -----

#layout_row_major = #iree_vector_ext.nested_layout<
  subgroups_per_workgroup = [1, 1],
  batches_per_subgroup    = [2, 2],
  outers_per_batch        = [1, 1],
  threads_per_outer       = [8, 1],
  elements_per_thread     = [1, 8],

  batch_order             = [1, 0],

  subgroup_basis          = [1, 1],
  thread_basis            = [8, 1]
>

// CHECK-DAG: #[[$MAP0:.+]] = affine_map<()[s0, s1] -> (s0 + s1)>
// CHECK-DAG: #[[$MAP1:.+]] = affine_map<(d0, d1, d2, d3) -> (d3, d2)>
// CHECK-DAG: #[[$MAP2:.+]] = affine_map<()[s0] -> (s0 + 8)>
// CHECK-DAG: #[[$MAP3:.+]] = affine_map<()[s0, s1] -> (s0 + s1 + 8)>

func.func @distribute_transfer_read_row_major_transpose(%a: index, %b: index, %arg0: memref<32x32x32x32xf16>) -> vector<16x16xf16> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f16
  %root = vector.transfer_read %arg0[%c0, %c0, %a, %b], %cst
          {in_bounds = [true, true],
           permutation_map = affine_map<(d0, d1, d2, d3) -> (d3, d2)>,
           "__vector_layout_test_anchor_result_0" = #layout_row_major}
                  : memref<32x32x32x32xf16>, vector<16x16xf16>
  func.return %root : vector<16x16xf16>
}

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: @distribute_transfer_read_row_major_transpose
// CHECK-SAME:    %[[I0:.+]]: index, %[[I1:.+]]: index

// CHECK: %[[IDS:.+]]:4 = affine.delinearize_index %{{.*}} into (%c1, %c1, %c8, %c1) : index, index, index, index
// CHECK: %[[LIN_ID0:.+]] = affine.apply #[[$MAP:.+]]()[%[[IDS]]#2, %[[I1]]]
// CHECK: vector.transfer_read %{{.*}}[%c0, %c0, %[[I0]], %[[LIN_ID0]]], {{.*}} permutation_map = #[[$MAP1]]
// CHECK: %[[I0_PLUS_8:.+]] = affine.apply #[[$MAP2]]()[%[[I0]]]
// CHECK: vector.transfer_read %{{.*}}[%c0, %c0, %[[I0_PLUS_8]], %[[LIN_ID0]]], {{.*}} permutation_map = #[[$MAP1]]
// CHECK: %[[LIN_ID1:.+]] = affine.apply #[[$MAP3]]()[%[[IDS]]#2, %[[I1]]]
// CHECK: vector.transfer_read %{{.*}}[%c0, %c0, %[[I0]], %[[LIN_ID1]]], {{.*}} permutation_map = #[[$MAP1]]
// CHECK: vector.transfer_read %{{.*}}[%c0, %c0, %[[I0_PLUS_8]], %[[LIN_ID1]]], %cst_0 {in_bounds = [true, true], permutation_map = #map1} : memref<32x32x32x32xf16>, vector<1x8xf16>

// -----

#layout_col_major = #iree_vector_ext.nested_layout<
  subgroups_per_workgroup = [1, 1],
  batches_per_subgroup    = [1, 2],
  outers_per_batch        = [1, 1],
  threads_per_outer       = [4, 8],
  elements_per_thread     = [4, 1],

  subgroup_basis          = [1, 1],
  thread_basis            = [4, 8]
>

// CHECK: #[[$MAP2:.+]] = affine_map<(d0, d1, d2, d3) -> (d3, d2)>

// CHECK-LABEL: @distribute_transfer_read_col_major_transpose
func.func @distribute_transfer_read_col_major_transpose(%a: index, %b: index, %arg0: memref<32x32x32x32xf16>) -> vector<16x16xf16> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f16
  %root = vector.transfer_read %arg0[%c0, %c0, %a, %b], %cst
          {in_bounds = [true, true],
           permutation_map = affine_map<(d0, d1, d2, d3) -> (d3, d2)>,
           "__vector_layout_test_anchor_result_0" = #layout_col_major}
                  : memref<32x32x32x32xf16>, vector<16x16xf16>
  func.return %root : vector<16x16xf16>
}

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}

// CHECK: vector.transfer_read {{.*}} permutation_map = #[[$MAP2]]
// CHECK: vector.transfer_read {{.*}} permutation_map = #[[$MAP2]]

// -----

#layout = #iree_vector_ext.nested_layout<
  subgroups_per_workgroup = [7, 3, 1, 1],
  batches_per_subgroup    = [3, 5, 2, 1],
  outers_per_batch        = [1, 1, 2, 4],
  threads_per_outer       = [1, 1, 2, 2],
  elements_per_thread     = [1, 1, 1, 2],

  subgroup_order          = [1, 0, 2, 3],
  batch_order             = [1, 2, 3, 0],
  outer_order             = [0, 3, 1, 2],
  thread_order            = [0, 1, 3, 2],

  subgroup_basis          = [7, 3, 1, 1],
  thread_basis            = [1, 1, 2, 2]
>

func.func @distribute_transfer_read_row_major_with_permutations(%a: index, %b: index, %arg0: memref<32x32x32x32xf16>) -> vector<21x15x8x16xf16> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f16
  %root = vector.transfer_read %arg0[%c0, %c0, %a, %b], %cst
          {in_bounds = [true, true, true, true],
           permutation_map = affine_map<(d0, d1, d2, d3) -> (d0, d3, 0, d1)>,
           "__vector_layout_test_anchor_result_0" = #layout}
                  : memref<32x32x32x32xf16>, vector<21x15x8x16xf16>
  func.return %root : vector<21x15x8x16xf16>
}

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}
// CHECK-LABEL: @distribute_transfer_read_row_major_with_permutations

// Verify that there are (batch0: 3) * (batch1: 5) * (outer3: 4) = 60 total
// unique transfer read ops. The broadcasted dimension (2) CSEs the duplicate
// reads.
// CHECK-COUNT-60: vector.transfer_read

// -----

#layout = #iree_vector_ext.nested_layout<
  subgroups_per_workgroup = [1],
  batches_per_subgroup    = [1],
  outers_per_batch        = [1],
  threads_per_outer       = [4],
  elements_per_thread     = [4],

  subgroup_basis          = [1],
  thread_basis            = [4, 16],
  thread_active_ids       = [true, false]
>

// CHECK-DAG: #[[$MAP:.+]] = affine_map<()[s0] -> (s0 * 4)>

// CHECK-LABEL: @distribute_transfer_read_broadcast
func.func @distribute_transfer_read_broadcast(%arg0: memref<32x32xf16>) -> vector<16xf16> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f16
  %root = vector.transfer_read %arg0[%c0, %c0], %cst
          {in_bounds = [true],
           "__vector_layout_test_anchor_result_0" = #layout}
                  : memref<32x32xf16>, vector<16xf16>
  func.return %root : vector<16xf16>
}

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}

// CHECK: %[[IDS:.+]]:3 = affine.delinearize_index %{{.*}} into (%c1, %c4, %c16) : index, index, index
// CHECK: %[[LANEY:.+]] = affine.apply #[[$MAP]]()[%[[IDS]]#1]
// CHECK: %[[RD:.+]] = vector.transfer_read %{{.*}}[%c0, %[[LANEY:.+]]], {{.*}} : memref<32x32xf16>, vector<4xf16>

// -----

#layout_row_major = #iree_vector_ext.nested_layout<
  subgroups_per_workgroup = [1, 1],
  batches_per_subgroup    = [2, 2],
  outers_per_batch        = [1, 1],
  threads_per_outer       = [8, 1],
  elements_per_thread     = [1, 8],

  batch_order             = [1, 0],

  subgroup_basis          = [1, 1],
  thread_basis            = [8, 1]
>

// CHECK: #[[$MAP:.+]] = affine_map<()[s0] -> (s0 + 8)>

// CHECK-LABEL: @distribute_transfer_write_row_major
func.func @distribute_transfer_write_row_major(%root: vector<16x16xf16>, %alloc: memref<64x64xf16>) {
  %c0 = arith.constant 0 : index
  vector.transfer_write %root, %alloc[%c0, %c0]
          {in_bounds = [true, true],
           "__vector_layout_test_anchor_operand_0" = #layout_row_major}
                  : vector<16x16xf16>, memref<64x64xf16>
  func.return
}

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}

// CHECK: %[[IDS:.+]]:4 = affine.delinearize_index %{{.*}} into (%c1, %c1, %c8, %c1) : index, index, index, index
// CHECK: %[[SLICE:.+]] = vector.extract %{{.*}}[0, 0, 0, 0] : vector<1x8xf16> from vector<2x2x1x1x1x8xf16>
// CHECK: vector.transfer_write %[[SLICE]], %{{.*}}[%[[IDS]]#2, %c0] {in_bounds = [true, true]} : vector<1x8xf16>, memref<64x64xf16>
// CHECK: vector.extract %{{.*}}[1, 0, 0, 0]
// CHECK: vector.transfer_write %{{.*}}, %{{.*}}[%[[IDS]]#2, %c8]
// CHECK: %[[LANEX_PLUS_VECDIMX:.+]] = affine.apply #[[$MAP]]()[%[[IDS]]#2]
// CHECK: vector.extract %{{.*}}[0, 1, 0, 0]
// CHECK: vector.transfer_write %{{.*}}[%[[LANEX_PLUS_VECDIMX]], %c0]
// CHECK: vector.extract %{{.*}}[1, 1, 0, 0]
// CHECK: vector.transfer_write %{{.*}}[%[[LANEX_PLUS_VECDIMX]], %c8]

// -----

#layout_col_major = #iree_vector_ext.nested_layout<
  subgroups_per_workgroup = [1, 1],
  batches_per_subgroup    = [1, 2],
  outers_per_batch        = [1, 1],
  threads_per_outer       = [4, 8],
  elements_per_thread     = [4, 1],

  batch_order             = [1, 0],
  element_order           = [1, 0],

  subgroup_basis          = [1, 1],
  thread_basis            = [4, 8]
>

// CHECK-DAG: #[[$MAP:.+]] = affine_map<()[s0] -> (s0 * 4)>
// CHECK-DAG: #[[$MAP1:.+]] = affine_map<()[s0] -> (s0 + 8)>

// CHECK-LABEL: @distribute_transfer_write_col_major
func.func @distribute_transfer_write_col_major(%root: vector<16x16xf16>, %alloc: memref<64x64xf16>) {
  %c0 = arith.constant 0 : index
  vector.transfer_write %root, %alloc[%c0, %c0]
          {in_bounds = [true, true],
           "__vector_layout_test_anchor_operand_0" = #layout_col_major}
                  : vector<16x16xf16>, memref<64x64xf16>
  func.return
}

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}

// CHECK: %[[IDS:.+]]:4 = affine.delinearize_index %0 into (%c1, %c1, %c4, %c8) : index, index, index, index
// CHECK: %[[LANEY:.+]] = affine.apply #map()[%1#2]
// CHECK: vector.extract %{{.*}}[0, 0, 0, 0]
// CHECK: vector.transpose %{{.*}}, [1, 0] : vector<1x4xf16> to vector<4x1xf16>
// CHECK: vector.transfer_write %{{.*}}[%[[LANEY]], %[[IDS]]#3]
// CHECK: %[[LANEX:.+]] = affine.apply #[[$MAP1]]()[%[[IDS]]#3]
// CHECK: vector.extract %{{.*}}[1, 0, 0, 0]
// CHECK: vector.transpose %{{.*}}, [1, 0] : vector<1x4xf16> to vector<4x1xf16>
// CHECK: vector.transfer_write {{.*}}[%[[LANEY]], %[[LANEX]]]

// -----

#layout_row_major = #iree_vector_ext.nested_layout<
  subgroups_per_workgroup = [1, 1],
  batches_per_subgroup    = [2, 2],
  outers_per_batch        = [1, 1],
  threads_per_outer       = [8, 1],
  elements_per_thread     = [1, 8],

  batch_order             = [1, 0],

  subgroup_basis          = [1, 1],
  thread_basis            = [8, 1]
>

// CHECK-DAG: #[[$MAP:.+]] = affine_map<()[s0, s1] -> (s0 + s1)>
// CHECK-DAG: #[[$MAP1:.+]] = affine_map<(d0, d1, d2, d3) -> (d3, d2)>
// CHECK-DAG: #[[$MAP2:.+]] = affine_map<()[s0] -> (s0 + 8)>
// CHECK-DAG: #[[$MAP3:.+]] = affine_map<()[s0, s1] -> (s0 + s1 + 8)>

func.func @distribute_transfer_write_row_major_with_nontrivial_index(%root: vector<16x16xf16>, %a: index, %b: index, %alloc: memref<32x32x32x32xf16>) {
  %c0 = arith.constant 0 : index
  vector.transfer_write %root, %alloc[%c0, %c0, %a, %b]
          {in_bounds = [true, true],
           permutation_map = affine_map<(d0, d1, d2, d3) -> (d3, d2)>,
           "__vector_layout_test_anchor_operand_0" = #layout_row_major}
                  : vector<16x16xf16>, memref<32x32x32x32xf16>
  func.return
}

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: @distribute_transfer_write_row_major_with_nontrivial_index
// CHECK-SAME:    vector<16x16xf16>, %[[I0:.+]]: index, %[[I1:.+]]: index

// CHECK: %[[IDS:.+]]:4 = affine.delinearize_index %{{.*}} into (%c1, %c1, %c8, %c1) : index, index, index, index
// CHECK: %[[LIN_ID0:.+]] = affine.apply #[[$MAP]]()[%[[IDS]]#2, %[[I1]]]
// CHECK: vector.extract %{{.*}}[0, 0, 0, 0]
// CHECK: vector.transfer_write %{{.*}}[%c0, %c0, %[[I0]], %[[LIN_ID0]]] {{.*}} permutation_map = #[[$MAP1]]
// CHECK: %[[LIN_ID1:.+]] = affine.apply #[[$MAP2]]()[%[[I0]]]
// CHECK: vector.extract %{{.*}}[1, 0, 0, 0]
// CHECK: vector.transfer_write %{{.*}}[%c0, %c0, %[[LIN_ID1]], %3] {{.*}} permutation_map = #[[$MAP1]]
// CHECK: %[[LIN_ID2:.+]] = affine.apply #[[$MAP3]]()[%[[IDS]]#2, %[[I1]]]
// CHECK: vector.extract %{{.*}}[0, 1, 0, 0]
// CHECK: vector.transfer_write %{{.*}}[%c0, %c0, %[[I0]], %[[LIN_ID2]]] {{.*}} permutation_map = #[[$MAP1]]
// CHECK: vector.extract %{{.*}}[1, 1, 0, 0]
// CHECK: vector.transfer_write %{{.*}}[%c0, %c0, %[[LIN_ID1]], %[[LIN_ID2]]] {{.*}} permutation_map = #[[$MAP1]]

// -----

#layout_row_major = #iree_vector_ext.nested_layout<
  subgroups_per_workgroup = [1, 1],
  batches_per_subgroup    = [2, 2],
  outers_per_batch        = [1, 1],
  threads_per_outer       = [8, 1],
  elements_per_thread     = [1, 8],

  batch_order             = [1, 0],

  subgroup_basis          = [1, 1],
  thread_basis            = [8, 1]
>

func.func @distribute_transfer_read_write(%a: index, %b: index,
                                          %arg0: memref<32x32x32x32xf16>,
                                          %arg1: memref<32x32x32x32xf16>) {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f16
  %root = vector.transfer_read %arg0[%c0, %c0, %a, %b], %cst
          {in_bounds = [true, true],
           permutation_map = affine_map<(d0, d1, d2, d3) -> (d2, d3)>,
           "__vector_layout_test_anchor_result_0" = #layout_row_major}
                  : memref<32x32x32x32xf16>, vector<16x16xf16>
  vector.transfer_write %root, %arg1[%c0, %c0, %a, %b]
          {in_bounds = [true, true],
           permutation_map = affine_map<(d0, d1, d2, d3) -> (d2, d3)>,
           "__vector_layout_test_anchor_operand_0" = #layout_row_major}
                  : vector<16x16xf16>, memref<32x32x32x32xf16>
  return
}

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}

// CHECK: %[[B00:.+]] = vector.transfer_read %{{.*}}[%c0, %c0, %[[LANEX:[a-zA-Z0-9]+]], %[[OFFSET0:[a-zA-Z0-9]+]]]
// CHECK: %[[B10:.+]] = vector.transfer_read %{{.*}}[%c0, %c0, %[[LANEX]], %[[OFFSET1:[a-zA-Z0-9]+]]]
// CHECK: %[[B01:.+]] = vector.transfer_read %{{.*}}[%c0, %c0, %[[LANEX_PLUS_BATCH:[a-zA-Z0-9]+]], %[[OFFSET0]]]
// CHECK: %[[B11:.+]] = vector.transfer_read %{{.*}}[%c0, %c0, %[[LANEX_PLUS_BATCH]], %[[OFFSET1]]]
// CHECK: vector.transfer_write %[[B00]], %{{.*}}[%c0, %c0, %[[LANEX]], %[[OFFSET0]]]
// CHECK: vector.transfer_write %[[B10]], %{{.*}}[%c0, %c0, %[[LANEX]], %[[OFFSET1]]]
// CHECK: vector.transfer_write %[[B01]], %{{.*}}[%c0, %c0, %[[LANEX_PLUS_BATCH]], %[[OFFSET0]]]
// CHECK: vector.transfer_write %[[B11]], %{{.*}}[%c0, %c0, %[[LANEX_PLUS_BATCH]], %[[OFFSET1]]]

// -----

// A: shape = 128x8, layout = layoutA
#layout_a = #iree_vector_ext.nested_layout<
  subgroups_per_workgroup = [4, 1],
  batches_per_subgroup    = [1, 1],
  outers_per_batch        = [1, 1],
  threads_per_outer       = [32, 2],
  elements_per_thread     = [1, 4],

  thread_order            = [1, 0],

  subgroup_basis          = [4, 2],
  thread_basis            = [2, 32]
>

// B: shape = 8x64, layout = layoutB
#layout_b = #iree_vector_ext.nested_layout<
  subgroups_per_workgroup = [1, 2],
  batches_per_subgroup    = [1, 1],
  outers_per_batch        = [1, 1],
  threads_per_outer       = [2, 32],
  elements_per_thread     = [4, 1],

  element_order           = [1, 0],

  subgroup_basis          = [4, 2],
  thread_basis            = [2, 32]
>

// C: shape = 128x64, layout = layoutC
#layout_c = #iree_vector_ext.nested_layout<
  subgroups_per_workgroup = [4, 2],
  batches_per_subgroup    = [1, 1],
  outers_per_batch        = [4, 1],
  threads_per_outer       = [2, 32],
  elements_per_thread     = [4, 1],

  element_order           = [1, 0],

  subgroup_basis          = [4, 2],
  thread_basis            = [2, 32]
>

// CHECK-DAG: #[[$MAP:.+]] = affine_map<()[s0, s1] -> (s0 * 32 + s1)>
// CHECK-DAG: #[[$MAP1:.+]] = affine_map<()[s0] -> (s0 * 4)>
// CHECK-DAG: #[[$MAP2:.+]] = affine_map<()[s0, s1] -> (s0 * 32 + s1 * 4)>
// CHECK-DAG: #[[$MAP3:.+]] = affine_map<()[s0, s1] -> (s0 * 32 + s1 * 4 + 8)>
// CHECK-DAG: #[[$MAP4:.+]] = affine_map<()[s0, s1] -> (s0 * 32 + s1 * 4 + 16)>
// CHECK-DAG: #[[$MAP5:.+]] = affine_map<()[s0, s1] -> (s0 * 32 + s1 * 4 + 24)>
// CHECK-LABEL: @mfma_64x128x8_read
func.func @mfma_64x128x8_read(%mem: memref<128x8xf16>,
                              %mem1: memref<8x64xf16>,
                              %mem2: memref<128x64xf16>)
                -> (vector<128x8xf16>, vector<8x64xf16>, vector<128x64xf16>) {

  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f16

  // CHECK: %[[IDX:.+]] = gpu.thread_id  x
  // CHECK: %[[IDS:.+]]:4 = affine.delinearize_index %[[IDX]] into (%c4, %c2, %c2, %c32) : index, index, index, index

  // CHECK-DAG: %[[M:.+]] = affine.apply #[[$MAP]]()[%[[IDS]]#0, %[[IDS]]#3]
  // CHECK-DAG: %[[N:.+]] = affine.apply #[[$MAP]]()[%[[IDS]]#1, %[[IDS]]#3]
  // CHECK-DAG: %[[K:.+]] = affine.apply #[[$MAP1]]()[%[[IDS]]#2]

  // M is unrolled 4 times.
  // CHECK-DAG: %[[M0:.+]] = affine.apply #[[$MAP2]]()[%[[IDS]]#0, %[[IDS]]#2]
  // CHECK-DAG: %[[M1:.+]] = affine.apply #[[$MAP3]]()[%[[IDS]]#0, %[[IDS]]#2]
  // CHECK-DAG: %[[M2:.+]] = affine.apply #[[$MAP4]]()[%[[IDS]]#0, %[[IDS]]#2]
  // CHECK-DAG: %[[M3:.+]] = affine.apply #[[$MAP5]]()[%[[IDS]]#0, %[[IDS]]#2]

  // M, K
  // CHECK-DAG: transfer_read %{{.*}}[%[[M]], %[[K]]]
  // K, N
  // CHECK-DAG: transfer_read %{{.*}}[%[[K]], %[[N]]]
  // M, N
  // CHECK-DAG: transfer_read %{{.*}}[%[[M0]], %[[N]]]
  // CHECK-DAG: transfer_read %{{.*}}[%[[M1]], %[[N]]]
  // CHECK-DAG: transfer_read %{{.*}}[%[[M2]], %[[N]]]
  // CHECK-DAG: transfer_read %{{.*}}[%[[M3]], %[[N]]

  %a = vector.transfer_read %mem[%c0, %c0], %cst
          {in_bounds = [true, true],
           "__vector_layout_test_anchor_result_0" = #layout_a}
  : memref<128x8xf16>, vector<128x8xf16>
  %b = vector.transfer_read %mem1[%c0, %c0], %cst
          {in_bounds = [true, true],
           "__vector_layout_test_anchor_result_0" = #layout_b}
  : memref<8x64xf16>, vector<8x64xf16>
  %c = vector.transfer_read %mem2[%c0, %c0], %cst
          {in_bounds = [true, true],
           "__vector_layout_test_anchor_result_0" = #layout_c}
  : memref<128x64xf16>, vector<128x64xf16>

  return %a, %b, %c : vector<128x8xf16>, vector<8x64xf16>, vector<128x64xf16>
}

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}

// -----

#layout = #iree_vector_ext.nested_layout<
  subgroups_per_workgroup = [2, 1],
  batches_per_subgroup    = [1, 1],
  outers_per_batch        = [1, 1],
  threads_per_outer       = [32, 2],
  elements_per_thread     = [1, 4],

  subgroup_order          = [1, 0],
  batch_order             = [1, 0],
  outer_order             = [1, 0],
  thread_order            = [1, 0],
  element_order           = [0, 1],

  subgroup_basis          = [4, 2],
  thread_basis            = [2, 32]
>

func.func @transposed_read_64x8(%mem: memref<8x64xf16>)
                -> (vector<64x8xf16>) {

  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f16

  %read = vector.transfer_read %mem[%c0, %c0], %cst
          {in_bounds = [true, true], permutation_map = affine_map<(d0, d1) -> (d1, d0)>,
           "__vector_layout_test_anchor_result_0" = #layout}
  : memref<8x64xf16>, vector<64x8xf16>

  return %read : vector<64x8xf16>
}

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}

// CHECK-DAG: #[[$MAP:.+]] = affine_map<()[s0, s1] -> (s0 * 32 + s1)>
// CHECK-DAG: #[[$MAP1:.+]] = affine_map<()[s0] -> (s0 * 4)>

// CHECK-LABEL: @transposed_read_64x8

// CHECK: %[[IDX:.+]] = gpu.thread_id  x
// Delinearization basis comes directly from the
// CHECK: %[[IDS:.+]]:4 = affine.delinearize_index %[[IDX]] into (%c4, %c2, %c2, %c32) : index, index, index, index

// CHECK-DAG: %[[M:.+]] = affine.apply #[[$MAP]]()[%[[IDS]]#1, %[[IDS]]#3]
// CHECK-DAG: %[[N:.+]] = affine.apply #[[$MAP1]]()[%[[IDS]]#2]
// CHECK: transfer_read %{{.*}}[%[[N]], %[[M]]

// -----

#layout = #iree_vector_ext.nested_layout<
  subgroups_per_workgroup = [2, 2],
  batches_per_subgroup = [2, 4],
  outers_per_batch = [1, 1],
  threads_per_outer = [4, 16],
  elements_per_thread = [4, 1],
  element_order = [1, 0],
  subgroup_basis = [2, 2],
  thread_basis = [4, 16]
>

func.func @broadcast(%src: vector<128xf16>) -> (vector<64x128xf16>) {
  %bcast = vector.broadcast %src {"__vector_layout_test_anchor_result_0" = #layout}
    : vector<128xf16> to vector<64x128xf16>
  return %bcast : vector<64x128xf16>
}

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}

// CHECK: vector.extract {{.*}}[0, 0] : vector<1xf16> from vector<4x1x1xf16>
// CHECK: vector.broadcast {{.*}} : vector<1xf16> to vector<1x4xf16>
// CHECK: vector.insert {{.*}} [0, 0, 0, 0] : vector<1x4xf16> into vector<2x4x1x1x1x4xf16>
// CHECK: vector.extract {{.*}}[1, 0] : vector<1xf16> from vector<4x1x1xf16>
// CHECK: vector.broadcast {{.*}} : vector<1xf16> to vector<1x4xf16>
// CHECK: vector.insert {{.*}} [0, 1, 0, 0] : vector<1x4xf16> into vector<2x4x1x1x1x4xf16>
// CHECK: vector.extract {{.*}}[2, 0] : vector<1xf16> from vector<4x1x1xf16>
// CHECK: vector.broadcast {{.*}} : vector<1xf16> to vector<1x4xf16>
// CHECK: vector.insert {{.*}} [0, 2, 0, 0] : vector<1x4xf16> into vector<2x4x1x1x1x4xf16>
// CHECK: vector.extract {{.*}}[3, 0] : vector<1xf16> from vector<4x1x1xf16>
// CHECK: vector.broadcast {{.*}} : vector<1xf16> to vector<1x4xf16>

// CHECK: vector.insert {{.*}} [1, 0, 0, 0] : vector<1x4xf16> into vector<2x4x1x1x1x4xf16>
// CHECK: vector.insert {{.*}} [1, 1, 0, 0] : vector<1x4xf16> into vector<2x4x1x1x1x4xf16>
// CHECK: vector.insert {{.*}} [1, 2, 0, 0] : vector<1x4xf16> into vector<2x4x1x1x1x4xf16>
// CHECK: vector.insert {{.*}} [1, 3, 0, 0] : vector<1x4xf16> into vector<2x4x1x1x1x4xf16>

// -----

#layout = #iree_vector_ext.nested_layout<
  subgroups_per_workgroup = [2, 2, 2],
  batches_per_subgroup = [2, 2, 1],
  outers_per_batch = [2, 1, 1],
  threads_per_outer = [4, 16, 8],
  elements_per_thread = [1, 4, 4],
  batch_order = [2, 1, 0],
  subgroup_basis = [2, 2, 2],
  thread_basis = [4, 16, 8]
>

func.func @broadcast(%src: vector<64xf16>) -> (vector<32x256x64xf16>) {
  %bcast = vector.broadcast %src {"__vector_layout_test_anchor_result_0" = #layout}
    : vector<64xf16> to vector<32x256x64xf16>
  return %bcast : vector<32x256x64xf16>
}

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}

// CHECK: %[[EXTRACT:.+]] = vector.extract %{{.*}}[0, 0] : vector<4xf16> from vector<1x1x4xf16>
// CHECK: %[[BCAST:.+]] = vector.broadcast %[[EXTRACT]] : vector<4xf16> to vector<1x4x4xf16>
// CHECK: vector.insert %[[BCAST]], %{{.*}} [0, 0, 0, 0, 0, 0] : vector<1x4x4xf16> into vector<1x2x2x2x1x1x1x4x4xf16>
// CHECK: vector.insert %[[BCAST]], %{{.*}} [0, 0, 0, 1, 0, 0] : vector<1x4x4xf16> into vector<1x2x2x2x1x1x1x4x4xf16>
// CHECK: vector.insert %[[BCAST]], %{{.*}} [0, 1, 0, 0, 0, 0] : vector<1x4x4xf16> into vector<1x2x2x2x1x1x1x4x4xf16>
// CHECK: vector.insert %[[BCAST]], %{{.*}} [0, 1, 0, 1, 0, 0] : vector<1x4x4xf16> into vector<1x2x2x2x1x1x1x4x4xf16>
// CHECK: vector.insert %[[BCAST]], %{{.*}} [0, 0, 1, 0, 0, 0] : vector<1x4x4xf16> into vector<1x2x2x2x1x1x1x4x4xf16>
// CHECK: vector.insert %[[BCAST]], %{{.*}} [0, 0, 1, 1, 0, 0] : vector<1x4x4xf16> into vector<1x2x2x2x1x1x1x4x4xf16>
// CHECK: vector.insert %[[BCAST]], %{{.*}} [0, 1, 1, 0, 0, 0] : vector<1x4x4xf16> into vector<1x2x2x2x1x1x1x4x4xf16>
// CHECK: vector.insert %[[BCAST]], %{{.*}} [0, 1, 1, 1, 0, 0] : vector<1x4x4xf16> into vector<1x2x2x2x1x1x1x4x4xf16>
