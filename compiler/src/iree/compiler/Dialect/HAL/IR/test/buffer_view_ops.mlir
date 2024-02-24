// RUN: iree-opt --split-input-file %s | iree-opt --split-input-file | FileCheck %s

// CHECK-LABEL: @element_type
util.func public @element_type() -> i32 {
  // CHECK: %[[RET:.+]] = hal.element_type<f32> : i32
  %element_type = hal.element_type<f32> : i32
  // CHECK: util.return %[[RET]]
  util.return %element_type : i32
}

// -----

// CHECK-LABEL: @encoding_type
util.func public @encoding_type() -> i32 {
  // CHECK: %[[RET:.+]] = hal.encoding_type<dense_row_major> : i32
  %encoding_type = hal.encoding_type<dense_row_major> : i32
  // CHECK: util.return %[[RET]]
  util.return %encoding_type : i32
}

// -----

// CHECK-LABEL: @buffer_view_create
util.func public @buffer_view_create(%arg0: !hal.buffer, %arg1: index, %arg2: index) -> !hal.buffer_view {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c1_i32 = arith.constant 1 : i32
  %c32_i32 = arith.constant 32 : i32
  // CHECK: %view = hal.buffer_view.create
  // CHECK-SAME: buffer(%arg0 : !hal.buffer)[%c0, %c128]
  // CHECK-SAME: shape([%arg1, %arg2])
  // CHECK-SAME: type(%c32_i32)
  // CHECK-SAME: encoding(%c1_i32) : !hal.buffer_view
  %view = hal.buffer_view.create buffer(%arg0 : !hal.buffer)[%c0, %c128]
                                 shape([%arg1, %arg2])
                                 type(%c32_i32)
                                 encoding(%c1_i32) : !hal.buffer_view
  util.return %view : !hal.buffer_view
}

// -----

// CHECK-LABEL: @buffer_view_buffer
util.func public @buffer_view_buffer(%arg0: !hal.buffer_view) -> !hal.buffer {
  // CHECK: %buffer = hal.buffer_view.buffer<%arg0 : !hal.buffer_view> : !hal.buffer
  %buffer = hal.buffer_view.buffer<%arg0 : !hal.buffer_view> : !hal.buffer
  util.return %buffer : !hal.buffer
}

// -----

// CHECK-LABEL: @buffer_view_shape_queries
util.func public @buffer_view_shape_queries(%arg0: !hal.buffer_view) -> (index, index) {
  // CHECK: %{{.+}} = hal.buffer_view.rank<%arg0 : !hal.buffer_view> : index
  %0 = hal.buffer_view.rank<%arg0 : !hal.buffer_view> : index
  // CHECK: %{{.+}} = hal.buffer_view.dim<%arg0 : !hal.buffer_view>[0] : index
  %1 = hal.buffer_view.dim<%arg0 : !hal.buffer_view>[0] : index
  util.return %0, %1 : index, index
}
