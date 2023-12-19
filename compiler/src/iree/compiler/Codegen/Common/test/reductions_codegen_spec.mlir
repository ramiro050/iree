// RUN: iree-opt %s

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    transform.iree.register_match_callbacks

    %maybe_leading, %original_fill, %reduction, %maybe_trailing_0 =
      transform.iree.match_callback failures(propagate) "reduction"(%root)
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)

    %_, %more_parallel_fill, %parallel_reduction, %combiner_op =
      transform.structured.split_reduction %reduction { split_factor = 2, insert_split_dimension = 1 }
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)

    // Step 1. Map to a single block by tiling with size 1 and fusing.
    %fusion_root_1, %fusion_group_1 = transform.iree.take_first %maybe_trailing_0, %combiner_op
      : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    %outer_tiled, %grid_loop = transform.structured.tile_using_forall %fusion_root_1 tile_sizes [1]
      ( mapping = [#gpu.block<x>] )
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    %func = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.iree.bubble_expand
    } : !transform.any_op

    // Excessively eager canonicalization results in `fill`s being "fused" due to
    // swapping with `extract_slice`, which confuses the fusion operation below.
    // Wrap fusion into a non-canonicalized sequence.
    %fused_2, %parallel_reduction_2, %more_parallel_fill_2, %original_fill_2, %maybe_leading_2 =
      transform.sequence %root : !transform.any_op -> !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op
      failures(propagate) {
    ^bb1(%arg1: !transform.any_op):
      %fused_22, %new_containing_1 = transform.structured.fuse_into_containing_op %fusion_group_1 into %grid_loop : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
      %parallel_reduction_22, %new_containing_2 = transform.structured.fuse_into_containing_op %parallel_reduction into %grid_loop : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
      %more_parallel_fill_22, %new_containing_3 = transform.structured.fuse_into_containing_op %more_parallel_fill into %grid_loop : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
      %original_fill_22, %new_containing_4 = transform.structured.fuse_into_containing_op %original_fill into %grid_loop : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
      %maybe_leading_22, %new_containing_5 = transform.structured.fuse_into_containing_op %maybe_leading into %grid_loop : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

      transform.yield %fused_22, %parallel_reduction_22, %more_parallel_fill_22, %original_fill_22, %maybe_leading_22
        : !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op
    }

    // Step 2. Map reduction to thread X and parallel dimension to other threads.
    // ===========================================================================
    %fusion_group_22_full = transform.merge_handles %fused_2, %original_fill_2
      : !transform.any_op
    %fusion_root_22_tiled, %block_loop_22 =
      transform.structured.tile_using_forall %outer_tiled
      tile_sizes [1] ( mapping = [#gpu.thread<z>] )
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.structured.fuse_into_containing_op %fusion_group_22_full into %block_loop_22 : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)


    %fusion_group_21 = transform.merge_handles %maybe_leading_2, %more_parallel_fill_2
      : !transform.any_op
    %fusion_root_21_tiled, %block_loop_21 =
      transform.structured.tile_using_forall %parallel_reduction_2
      tile_sizes [1, 1] ( mapping = [#gpu.thread<z>, #gpu.thread<y>] )
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.structured.fuse_into_containing_op %fusion_group_21 into %block_loop_21 : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Step 3. Rank-reduce.
    // ===========================================================================
    transform.apply_patterns to %func {
      transform.apply_patterns.iree.fold_reshape_into_tensor_hal_interface
      transform.apply_patterns.linalg.fold_unit_extent_dims_via_slices
      transform.apply_patterns.vector.cast_away_vector_leading_one_dim
    } : !transform.any_op

    // We don't perform any following transformation (vectorization, bufferizaton,
    // mapping) because this schedule is applied to Linalg-only code without the
    // surrounding context and because it would make it difficult to detect, e.g.,
    // lack of fusion.
    transform.yield
  } // @__transform_main
} // module
