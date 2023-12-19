// RUN: iree-opt %s

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    transform.iree.register_match_callbacks

    %leading, %fill, %reduction, %trailing =
      transform.iree.match_callback failures(propagate) "reduction"(%root)
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)

    transform.iree.emit_remark "leading" at %leading : !transform.any_op
    transform.iree.emit_remark "fill" at %fill : !transform.any_op
    transform.iree.emit_remark "reduction" at %reduction : !transform.any_op
    transform.iree.emit_remark "trailing" at %trailing : !transform.any_op
    transform.yield
  } // @__transform_main
} // module
