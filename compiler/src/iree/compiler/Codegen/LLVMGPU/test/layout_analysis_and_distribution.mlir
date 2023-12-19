// RUN: iree-opt %s -iree-transform-dialect-interpreter -transform-dialect-drop-schedule -cse -split-input-file --verify-diagnostics | FileCheck %s

builtin.module attributes { transform.with_named_sequence } {
  func.func @matmul_dispatch_0_matmul_16x8x16() {
    %c0 = arith.constant 0 : index
    %cst = arith.constant dense<0.000000e+00> : vector<16x8xf16>
    %cst_0 = arith.constant 0.000000e+00 : f16
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : memref<16x16xf16>
    memref.assume_alignment %0, 64 : memref<16x16xf16>
    %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : memref<8x16xf16>
    memref.assume_alignment %1, 64 : memref<8x16xf16>
    %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : memref<16x8xf16>
    memref.assume_alignment %2, 64 : memref<16x8xf16>
    %3 = vector.transfer_read %0[%c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<16x16xf16>, vector<16x16xf16>
    %4 = vector.transfer_read %1[%c0, %c0], %cst_0 {permutation_map = affine_map<(d0, d1) -> (d1, d0)>, in_bounds = [true, true]} : memref<8x16xf16>, vector<8x16xf16>
    %5 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %3, %4, %cst : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
    vector.transfer_write %5, %2[%c0, %c0] {in_bounds = [true, true]} : vector<16x8xf16>, memref<16x8xf16>
    return
  }

  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    %transformed_func = transform.iree.layout_analysis_and_distribution %top_level_func : (!transform.any_op) -> (!transform.any_op)
    transform.yield
  }
} // module


// CHECK-DAG:  #[[MAP:.+]] = affine_map<(d0, d1, d2) -> (d1 + d2 * 16)>
// CHECK-DAG:  #[[MAP1:.+]] = affine_map<(d0, d1, d2) -> (d0 * 2)>
// CHECK-DAG:  #[[MAP2:.+]] = affine_map<(d0, d1, d2) -> (d0 * 2 + 1)>
// CHECK-DAG:  #[[MAP3:.+]] = affine_map<(d0, d1, d2) -> (d0 * 2 + 8)>
// CHECK-DAG:  #[[MAP4:.+]] = affine_map<(d0, d1, d2) -> (d0 * 2 + 9)>
// CHECK-DAG:  #[[MAP5:.+]] = affine_map<(d0, d1, d2) -> (d1 + d2 * 16 + 8)>
// CHECK-DAG:  #[[MAP6:.+]] = affine_map<(d0, d1, d2) -> (d1 + d2 * 8)>
// CHECK:      func.func @matmul_dispatch_0_matmul_16x8x16() {
// CHECK-DAG:    %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:    %[[CST:.+]] = arith.constant dense<0.000000e+00> : vector<1x1x2x2xf16>
// CHECK:        %[[D0:.+]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64)
// CHECK-SAME:     offset(%[[C0]]) flags(ReadOnly) : memref<16x16xf16>
// CHECK:        memref.assume_alignment %[[D0]], 64 : memref<16x16xf16>
// CHECK:        %[[D1:.+]] = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64)
// CHECK-SAME:     offset(%[[C0]]) flags(ReadOnly) : memref<8x16xf16>
// CHECK:        memref.assume_alignment %[[D1]], 64 : memref<8x16xf16>
// CHECK:        %[[D2:.+]] = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64)
// CHECK-SAME:     offset(%[[C0]]) : memref<16x8xf16>
// CHECK:        memref.assume_alignment %[[D2]], 64 : memref<16x8xf16>
// CHECK-DAG:    %[[D3:.+]] = gpu.thread_id  x
// CHECK-DAG:    %[[D4:.+]] = gpu.thread_id  y
// CHECK-DAG:    %[[D5:.+]] = gpu.thread_id  z
// CHECK-DAG:    %[[CST_0:.+]] = arith.constant dense<0.000000e+00> : vector<1x1x4x2xf16>
// CHECK-DAG:    %[[D6:.+]] = affine.apply #[[MAP]](%[[D3]], %[[D4]], %[[D5]])
// CHECK-DAG:    %[[D7:.+]] = affine.apply #[[MAP1]](%[[D3]], %[[D4]], %[[D5]])
// CHECK:        %[[D8:.+]] = arith.addi %[[D6]], %[[C0]] : index
// CHECK:        %[[D9:.+]] = arith.addi %[[D7]], %[[C0]] : index
// CHECK:        %[[D10:.+]] = memref.load %[[D0]][%[[D8]], %[[D9]]] : memref<16x16xf16>
// CHECK:        %[[D11:.+]] = vector.broadcast %[[D10]] : f16 to vector<1xf16>
// CHECK:        %[[D12:.+]] = vector.insert_strided_slice %[[D11]], %[[CST_0]] {offsets = [0, 0, 0, 0], strides = [1]}
// CHECK-SAME:     : vector<1xf16> into vector<1x1x4x2xf16>
// CHECK-DAG:    %[[D13:.+]] = affine.apply #[[MAP2]](%[[D3]], %[[D4]], %[[D5]])
// CHECK:        %[[D14:.+]] = arith.addi %[[D13]], %[[C0]] : index
// CHECK:        %[[D15:.+]] = memref.load %[[D0]][%[[D8]], %[[D14]]] : memref<16x16xf16>
// CHECK:        %[[D16:.+]] = vector.broadcast %[[D15]] : f16 to vector<1xf16>
// CHECK:        %[[D17:.+]] = vector.insert_strided_slice %[[D16]], %[[D12]] {offsets = [0, 0, 0, 1], strides = [1]} :
// CHECK-SAME:     vector<1xf16> into vector<1x1x4x2xf16>
// CHECK-DAG:    %[[D18:.+]] = affine.apply #[[MAP3]](%[[D3]], %[[D4]], %[[D5]])
// CHECK:        %[[D19:.+]] = arith.addi %[[D18]], %[[C0]] : index
// CHECK:        %[[D20:.+]] = memref.load %[[D0]][%[[D8]], %[[D19]]] : memref<16x16xf16>
// CHECK:        %[[D21:.+]] = vector.broadcast %[[D20]] : f16 to vector<1xf16>
// CHECK:        %[[D22:.+]] = vector.insert_strided_slice %[[D21]], %[[D17]] {offsets = [0, 0, 2, 0], strides = [1]} :
// CHECK-SAME:     vector<1xf16> into vector<1x1x4x2xf16>
// CHECK-DAG:    %[[D23:.+]] = affine.apply #[[MAP4]](%[[D3]], %[[D4]], %[[D5]])
// CHECK:        %[[D24:.+]] = arith.addi %[[D23]], %[[C0]] : index
// CHECK:        %[[D25:.+]] = memref.load %[[D0]][%[[D8]], %[[D24]]] : memref<16x16xf16>
// CHECK:        %[[D26:.+]] = vector.broadcast %[[D25]] : f16 to vector<1xf16>
// CHECK:        %[[D27:.+]] = vector.insert_strided_slice %[[D26]], %[[D22]] {offsets = [0, 0, 2, 1], strides = [1]} :
// CHECK-SAME:     vector<1xf16> into vector<1x1x4x2xf16>
// CHECK-DAG:    %[[D28:.+]] = affine.apply #[[MAP5]](%[[D3]], %[[D4]], %[[D5]])
// CHECK:        %[[D29:.+]] = arith.addi %[[D28]], %[[C0]] : index
// CHECK:        %[[D30:.+]] = memref.load %[[D0]][%[[D29]], %[[D9]]] : memref<16x16xf16>
// CHECK:        %[[D31:.+]] = vector.broadcast %[[D30]] : f16 to vector<1xf16>
// CHECK:        %[[D32:.+]] = vector.insert_strided_slice %[[D31]], %[[D27]] {offsets = [0, 0, 1, 0], strides = [1]} :
// CHECK-SAME:     vector<1xf16> into vector<1x1x4x2xf16>
// CHECK:        %[[D33:.+]] = memref.load %[[D0]][%[[D29]], %[[D14]]] : memref<16x16xf16>
// CHECK:        %[[D34:.+]] = vector.broadcast %[[D33]] : f16 to vector<1xf16>
// CHECK:        %[[D35:.+]] = vector.insert_strided_slice %[[D34]], %[[D32]] {offsets = [0, 0, 1, 1], strides = [1]} :
// CHECK-SAME:     vector<1xf16> into vector<1x1x4x2xf16>
// CHECK:        %[[D36:.+]] = memref.load %[[D0]][%[[D29]], %[[D19]]] : memref<16x16xf16>
// CHECK:        %[[D37:.+]] = vector.broadcast %[[D36]] : f16 to vector<1xf16>
// CHECK:        %[[D38:.+]] = vector.insert_strided_slice %[[D37]], %[[D35]] {offsets = [0, 0, 3, 0], strides = [1]} :
// CHECK-SAME:     vector<1xf16> into vector<1x1x4x2xf16>
// CHECK:        %[[D39:.+]] = memref.load %[[D0]][%[[D29]], %[[D24]]] : memref<16x16xf16>
// CHECK:        %[[D40:.+]] = vector.broadcast %[[D39]] : f16 to vector<1xf16>
// CHECK:        %[[D41:.+]] = vector.insert_strided_slice %[[D40]], %[[D38]] {offsets = [0, 0, 3, 1], strides = [1]} :
// CHECK-SAME:     vector<1xf16> into vector<1x1x4x2xf16>
// CHECK-DAG:    %[[D42:.+]] = affine.apply #[[MAP6]](%[[D3]], %[[D4]], %[[D5]])
// CHECK:        %[[D43:.+]] = arith.addi %[[D42]], %[[C0]] : index
// CHECK:        %[[D44:.+]] = memref.load %[[D1]][%[[D9]], %[[D43]]] : memref<8x16xf16>
// CHECK:        %[[D45:.+]] = vector.broadcast %[[D44]] : f16 to vector<1xf16>
// CHECK:        %[[D46:.+]] = vector.insert_strided_slice %[[D45]], %[[CST]] {offsets = [0, 0, 0, 0], strides = [1]} :
// CHECK-SAME:     vector<1xf16> into vector<1x1x2x2xf16>
// CHECK:        %[[D47:.+]] = memref.load %[[D1]][%[[D14]], %[[D43]]] : memref<8x16xf16>
// CHECK:        %[[D48:.+]] = vector.broadcast %[[D47]] : f16 to vector<1xf16>
// CHECK:        %[[D49:.+]] = vector.insert_strided_slice %[[D48]], %[[D46]] {offsets = [0, 0, 0, 1], strides = [1]} :
// CHECK-SAME:     vector<1xf16> into vector<1x1x2x2xf16>
// CHECK:        %[[D50:.+]] = memref.load %[[D1]][%[[D19]], %[[D43]]] : memref<8x16xf16>
// CHECK:        %[[D51:.+]] = vector.broadcast %[[D50]] : f16 to vector<1xf16>
// CHECK:        %[[D52:.+]] = vector.insert_strided_slice %[[D51]], %[[D49]] {offsets = [0, 0, 1, 0], strides = [1]} :
// CHECK-SAME:     vector<1xf16> into vector<1x1x2x2xf16>
// CHECK:        %[[D53:.+]] = memref.load %[[D1]][%[[D24]], %[[D43]]] : memref<8x16xf16>
// CHECK:        %[[D54:.+]] = vector.broadcast %[[D53]] : f16 to vector<1xf16>
// CHECK:        %[[D55:.+]] = vector.insert_strided_slice %[[D54]], %[[D52]] {offsets = [0, 0, 1, 1], strides = [1]} :
// CHECK-SAME:     vector<1xf16> into vector<1x1x2x2xf16>
// CHECK:        %[[D56:.+]] = vector.extract %[[CST]][0, 0] : vector<2x2xf16> from vector<1x1x2x2xf16>
// CHECK:        %[[D57:.+]] = vector.extract %[[D41]][0, 0] : vector<4x2xf16> from vector<1x1x4x2xf16>
// CHECK:        %[[D58:.+]] = vector.extract %[[D55]][0, 0] : vector<2x2xf16> from vector<1x1x2x2xf16>
// CHECK:        %[[D59:.+]] = nvgpu.mma.sync(%[[D57]], %[[D58]], %[[D56]]) {mmaShape = [16, 8, 16]} : (vector<4x2xf16>,
// CHECK-SAME:     vector<2x2xf16>, vector<2x2xf16>) -> vector<2x2xf16>
// CHECK:        %[[D60:.+]] = vector.insert %[[D59]], %[[CST]] [0, 0] : vector<2x2xf16> into vector<1x1x2x2xf16>
// CHECK:        %[[D61:.+]] = vector.extract %[[D60]][0, 0, 0, 0] : f16 from vector<1x1x2x2xf16>
// CHECK:        memref.store %[[D61]], %[[D2]][%[[D8]], %[[D9]]] : memref<16x8xf16>
// CHECK:        %[[D62:.+]] = vector.extract %[[D60]][0, 0, 0, 1] : f16 from vector<1x1x2x2xf16>
// CHECK:        memref.store %[[D62]], %[[D2]][%[[D8]], %[[D14]]] : memref<16x8xf16>
// CHECK:        %[[D63:.+]] = vector.extract %[[D60]][0, 0, 1, 0] : f16 from vector<1x1x2x2xf16>
// CHECK:        memref.store %[[D63]], %[[D2]][%[[D29]], %[[D9]]] : memref<16x8xf16>
// CHECK:        %[[D64:.+]] = vector.extract %[[D60]][0, 0, 1, 1] : f16 from vector<1x1x2x2xf16>
// CHECK:        memref.store %[[D64]], %[[D2]][%[[D29]], %[[D14]]] : memref<16x8xf16>
// CHECK:        return
// CHECK:      }

// -----

builtin.module attributes { transform.with_named_sequence } {
  func.func @matmul_reduction() {
    %c0 = arith.constant 0 : index
    %cst = arith.constant dense<0.000000e+00> : vector<16x8xf16>
    %init = arith.constant dense<-1.000000e+04> : vector<16xf16>
    %cst_0 = arith.constant 0.000000e+00 : f16
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : memref<16x16xf16>
    memref.assume_alignment %0, 64 : memref<16x16xf16>
    %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : memref<8x16xf16>
    memref.assume_alignment %1, 64 : memref<8x16xf16>
    %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : memref<16x8xf16>
    memref.assume_alignment %2, 64 : memref<16x8xf16>
    %3 = vector.transfer_read %0[%c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<16x16xf16>, vector<16x16xf16>
    %4 = vector.transfer_read %1[%c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<8x16xf16>, vector<8x16xf16>
    %5 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %3, %4, %cst : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
    %6 = vector.multi_reduction <maximumf>, %5, %init [1] : vector<16x8xf16> to vector<16xf16>
    %7 = vector.broadcast %6 : vector<16xf16> to vector<8x16xf16>
    %8 = vector.transpose %7, [1, 0] : vector<8x16xf16> to vector<16x8xf16>
    vector.transfer_write %8, %2[%c0, %c0] {in_bounds = [true, true]} : vector<16x8xf16>, memref<16x8xf16>
    return
  }
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    %transformed_func = transform.iree.layout_analysis_and_distribution %top_level_func : (!transform.any_op) -> (!transform.any_op)
    transform.yield
  }
} // module


// CHECK-DAG:  #[[MAP:.+]] = affine_map<(d0, d1, d2) -> (d1 + d2 * 16)>
// CHECK-DAG:  #[[MAP1:.+]] = affine_map<(d0, d1, d2) -> (d0 * 2)>
// CHECK-DAG:  #[[MAP2:.+]] = affine_map<(d0, d1, d2) -> (d0 * 2 + 1)>
// CHECK-DAG:  #[[MAP3:.+]] = affine_map<(d0, d1, d2) -> (d0 * 2 + 8)>
// CHECK-DAG:  #[[MAP4:.+]] = affine_map<(d0, d1, d2) -> (d0 * 2 + 9)>
// CHECK-DAG:  #[[MAP5:.+]] = affine_map<(d0, d1, d2) -> (d1 + d2 * 16 + 8)>
// CHECK-DAG:  #[[MAP6:.+]] = affine_map<(d0, d1, d2) -> (d1 + d2 * 8)>
// CHECK:      func.func @matmul_reduction() {
// CHECK-DAG:    %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:    %[[CST:.+]] = arith.constant dense<0.000000e+00> : vector<1x1x2x2xf16>
// CHECK-DAG:    %[[CST_0:.+]] = arith.constant dense<-1.000000e+04> : vector<1x1x2x2xf16>
// CHECK:        %[[D0:.+]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64)
// CHECK-SAME:     offset(%[[C0]]) flags(ReadOnly) : memref<16x16xf16>
// CHECK:        memref.assume_alignment %[[D0]], 64 : memref<16x16xf16>
// CHECK:        %[[D1:.+]] = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64)
// CHECK-SAME:     offset(%[[C0]]) flags(ReadOnly) : memref<8x16xf16>
// CHECK:        memref.assume_alignment %[[D1]], 64 : memref<8x16xf16>
// CHECK:        %[[D2:.+]] = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64)
// CHECK-SAME:     offset(%[[C0]]) : memref<16x8xf16>
// CHECK:        memref.assume_alignment %[[D2]], 64 : memref<16x8xf16>
// CHECK-DAG:    %[[D3:.+]] = gpu.thread_id  x
// CHECK-DAG:    %[[D4:.+]] = gpu.thread_id  y
// CHECK-DAG:    %[[D5:.+]] = gpu.thread_id  z
// CHECK-DAG:    %[[CST_1:.+]] = arith.constant dense<0.000000e+00> : vector<1x1x4x2xf16>
// CHECK-DAG:    %[[D6:.+]] = affine.apply #[[MAP]](%[[D3]], %[[D4]], %[[D5]])
// CHECK-DAG:    %[[D7:.+]] = affine.apply #[[MAP1]](%[[D3]], %[[D4]], %[[D5]])
// CHECK:        %[[D8:.+]] = arith.addi %[[D6]], %[[C0]] : index
// CHECK:        %[[D9:.+]] = arith.addi %[[D7]], %[[C0]] : index
// CHECK:        %[[D10:.+]] = memref.load %[[D0]][%[[D8]], %[[D9]]] : memref<16x16xf16>
// CHECK:        %[[D11:.+]] = vector.broadcast %[[D10]] : f16 to vector<1xf16>
// CHECK:        %[[D12:.+]] = vector.insert_strided_slice %[[D11]], %[[CST_1]] {offsets = [0, 0, 0, 0], strides = [1]}
// CHECK-SAME:     : vector<1xf16> into vector<1x1x4x2xf16>
// CHECK-DAG:    %[[D13:.+]] = affine.apply #[[MAP2]](%[[D3]], %[[D4]], %[[D5]])
// CHECK:        %[[D14:.+]] = arith.addi %[[D13]], %[[C0]] : index
// CHECK:        %[[D15:.+]] = memref.load %[[D0]][%[[D8]], %[[D14]]] : memref<16x16xf16>
// CHECK:        %[[D16:.+]] = vector.broadcast %[[D15]] : f16 to vector<1xf16>
// CHECK:        %[[D17:.+]] = vector.insert_strided_slice %[[D16]], %[[D12]] {offsets = [0, 0, 0, 1], strides = [1]} :
// CHECK-SAME:     vector<1xf16> into vector<1x1x4x2xf16>
// CHECK-DAG:    %[[D18:.+]] = affine.apply #[[MAP3]](%[[D3]], %[[D4]], %[[D5]])
// CHECK:        %[[D19:.+]] = arith.addi %[[D18]], %[[C0]] : index
// CHECK:        %[[D20:.+]] = memref.load %[[D0]][%[[D8]], %[[D19]]] : memref<16x16xf16>
// CHECK:        %[[D21:.+]] = vector.broadcast %[[D20]] : f16 to vector<1xf16>
// CHECK:        %[[D22:.+]] = vector.insert_strided_slice %[[D21]], %[[D17]] {offsets = [0, 0, 2, 0], strides = [1]} :
// CHECK-SAME:     vector<1xf16> into vector<1x1x4x2xf16>
// CHECK-DAG:    %[[D23:.+]] = affine.apply #[[MAP4]](%[[D3]], %[[D4]], %[[D5]])
// CHECK:        %[[D24:.+]] = arith.addi %[[D23]], %[[C0]] : index
// CHECK:        %[[D25:.+]] = memref.load %[[D0]][%[[D8]], %[[D24]]] : memref<16x16xf16>
// CHECK:        %[[D26:.+]] = vector.broadcast %[[D25]] : f16 to vector<1xf16>
// CHECK:        %[[D27:.+]] = vector.insert_strided_slice %[[D26]], %[[D22]] {offsets = [0, 0, 2, 1], strides = [1]} :
// CHECK-SAME:     vector<1xf16> into vector<1x1x4x2xf16>
// CHECK-DAG:    %[[D28:.+]] = affine.apply #[[MAP5]](%[[D3]], %[[D4]], %[[D5]])
// CHECK:        %[[D29:.+]] = arith.addi %[[D28]], %[[C0]] : index
// CHECK:        %[[D30:.+]] = memref.load %[[D0]][%[[D29]], %[[D9]]] : memref<16x16xf16>
// CHECK:        %[[D31:.+]] = vector.broadcast %[[D30]] : f16 to vector<1xf16>
// CHECK:        %[[D32:.+]] = vector.insert_strided_slice %[[D31]], %[[D27]] {offsets = [0, 0, 1, 0], strides = [1]} :
// CHECK-SAME:     vector<1xf16> into vector<1x1x4x2xf16>
// CHECK:        %[[D33:.+]] = memref.load %[[D0]][%[[D29]], %[[D14]]] : memref<16x16xf16>
// CHECK:        %[[D34:.+]] = vector.broadcast %[[D33]] : f16 to vector<1xf16>
// CHECK:        %[[D35:.+]] = vector.insert_strided_slice %[[D34]], %[[D32]] {offsets = [0, 0, 1, 1], strides = [1]} :
// CHECK-SAME:     vector<1xf16> into vector<1x1x4x2xf16>
// CHECK:        %[[D36:.+]] = memref.load %[[D0]][%[[D29]], %[[D19]]] : memref<16x16xf16>
// CHECK:        %[[D37:.+]] = vector.broadcast %[[D36]] : f16 to vector<1xf16>
// CHECK:        %[[D38:.+]] = vector.insert_strided_slice %[[D37]], %[[D35]] {offsets = [0, 0, 3, 0], strides = [1]} :
// CHECK-SAME:     vector<1xf16> into vector<1x1x4x2xf16>
// CHECK:        %[[D39:.+]] = memref.load %[[D0]][%[[D29]], %[[D24]]] : memref<16x16xf16>
// CHECK:        %[[D40:.+]] = vector.broadcast %[[D39]] : f16 to vector<1xf16>
// CHECK:        %[[D41:.+]] = vector.insert_strided_slice %[[D40]], %[[D38]] {offsets = [0, 0, 3, 1], strides = [1]} :
// CHECK-SAME:     vector<1xf16> into vector<1x1x4x2xf16>
// CHECK-DAG:    %[[D42:.+]] = affine.apply #[[MAP6]](%[[D3]], %[[D4]], %[[D5]])
// CHECK:        %[[D43:.+]] = arith.addi %[[D42]], %[[C0]] : index
// CHECK:        %[[D44:.+]] = memref.load %[[D1]][%[[D43]], %[[D9]]] : memref<8x16xf16>
// CHECK:        %[[D45:.+]] = vector.broadcast %[[D44]] : f16 to vector<1xf16>
// CHECK:        %[[D46:.+]] = vector.insert_strided_slice %[[D45]], %[[CST]] {offsets = [0, 0, 0, 0], strides = [1]} :
// CHECK-SAME:     vector<1xf16> into vector<1x1x2x2xf16>
// CHECK:        %[[D47:.+]] = memref.load %[[D1]][%[[D43]], %[[D14]]] : memref<8x16xf16>
// CHECK:        %[[D48:.+]] = vector.broadcast %[[D47]] : f16 to vector<1xf16>
// CHECK:        %[[D49:.+]] = vector.insert_strided_slice %[[D48]], %[[D46]] {offsets = [0, 0, 0, 1], strides = [1]} :
// CHECK-SAME:     vector<1xf16> into vector<1x1x2x2xf16>
// CHECK:        %[[D50:.+]] = memref.load %[[D1]][%[[D43]], %[[D19]]] : memref<8x16xf16>
// CHECK:        %[[D51:.+]] = vector.broadcast %[[D50]] : f16 to vector<1xf16>
// CHECK:        %[[D52:.+]] = vector.insert_strided_slice %[[D51]], %[[D49]] {offsets = [0, 0, 1, 0], strides = [1]} :
// CHECK-SAME:     vector<1xf16> into vector<1x1x2x2xf16>
// CHECK:        %[[D53:.+]] = memref.load %[[D1]][%[[D43]], %[[D24]]] : memref<8x16xf16>
// CHECK:        %[[D54:.+]] = vector.broadcast %[[D53]] : f16 to vector<1xf16>
// CHECK:        %[[D55:.+]] = vector.insert_strided_slice %[[D54]], %[[D52]] {offsets = [0, 0, 1, 1], strides = [1]} :
// CHECK-SAME:     vector<1xf16> into vector<1x1x2x2xf16>
// CHECK:        %[[D56:.+]] = vector.extract %[[CST]][0, 0] : vector<2x2xf16> from vector<1x1x2x2xf16>
// CHECK:        %[[D57:.+]] = vector.extract %[[D41]][0, 0] : vector<4x2xf16> from vector<1x1x4x2xf16>
// CHECK:        %[[D58:.+]] = vector.extract %[[D55]][0, 0] : vector<2x2xf16> from vector<1x1x2x2xf16>
// CHECK:        %[[D59:.+]] = nvgpu.mma.sync(%[[D57]], %[[D58]], %[[D56]]) {mmaShape = [16, 8, 16]} : (vector<4x2xf16>,
// CHECK-SAME:     vector<2x2xf16>, vector<2x2xf16>) -> vector<2x2xf16>
// CHECK:        %[[D60:.+]] = vector.insert %[[D59]], %[[CST]] [0, 0] : vector<2x2xf16> into vector<1x1x2x2xf16>
// CHECK:        %[[D61:.+]] = vector.extract %[[CST_0]][0, 0, 0, 0] : f16 from vector<1x1x2x2xf16>
// CHECK-DAG:    %[[CST_2:.+]] = arith.constant dense<0.000000e+00> : vector<2xf16>
// CHECK:        %[[D62:.+]] = vector.extract %[[D60]][0, 0, 0, 0] : f16 from vector<1x1x2x2xf16>
// CHECK:        %[[D63:.+]] = vector.insert %[[D62]], %[[CST_2]] [0] : f16 into vector<2xf16>
// CHECK:        %[[D64:.+]] = vector.extract %[[D60]][0, 0, 0, 1] : f16 from vector<1x1x2x2xf16>
// CHECK:        %[[D65:.+]] = vector.insert %[[D64]], %[[D63]] [1] : f16 into vector<2xf16>
// CHECK:        %[[D66:.+]] = vector.bitcast %[[D65]] : vector<2xf16> to vector<1xi32>
// CHECK:        %[[D67:.+]] = vector.extract %[[D66]][0] : i32 from vector<1xi32>
// CHECK-DAG:    %[[C1_I32:.+]] = arith.constant 1 : i32
// CHECK-DAG:    %[[C32_I32:.+]] = arith.constant 32 : i32
// CHECK:        %[[SHUFFLERESULT:.+]], %[[VALID:.+]] = gpu.shuffle  xor %[[D67]], %[[C1_I32]], %[[C32_I32]] : i32
// CHECK:        %[[D68:.+]] = vector.broadcast %[[SHUFFLERESULT]] : i32 to vector<1xi32>
// CHECK:        %[[D69:.+]] = vector.bitcast %[[D68]] : vector<1xi32> to vector<2xf16>
// CHECK:        %[[D70:.+]] = arith.maximumf %[[D69]], %[[D65]] : vector<2xf16>
// CHECK:        %[[D71:.+]] = vector.bitcast %[[D70]] : vector<2xf16> to vector<1xi32>
// CHECK:        %[[D72:.+]] = vector.extract %[[D71]][0] : i32 from vector<1xi32>
// CHECK-DAG:    %[[C2_I32:.+]] = arith.constant 2 : i32
// CHECK:        %[[SHUFFLERESULT_3:.+]], %[[VALID_4:.+]] = gpu.shuffle  xor %[[D72]], %[[C2_I32]], %[[C32_I32]] : i32
// CHECK:        %[[D73:.+]] = vector.broadcast %[[SHUFFLERESULT_3]] : i32 to vector<1xi32>
// CHECK:        %[[D74:.+]] = vector.bitcast %[[D73]] : vector<1xi32> to vector<2xf16>
// CHECK:        %[[D75:.+]] = arith.maximumf %[[D74]], %[[D70]] : vector<2xf16>
// CHECK:        %[[D76:.+]] = vector.extract %[[D75]][0] : f16 from vector<2xf16>
// CHECK:        %[[D77:.+]] = vector.extract %[[D75]][1] : f16 from vector<2xf16>
// CHECK:        %[[D78:.+]] = arith.maximumf %[[D76]], %[[D77]] : f16
// CHECK:        %[[D79:.+]] = arith.maximumf %[[D78]], %[[D61]] : f16
// CHECK:        %[[D80:.+]] = vector.insert %[[D79]], %[[CST]] [0, 0, 0, 0] : f16 into vector<1x1x2x2xf16>
// CHECK:        %[[D81:.+]] = vector.insert %[[D79]], %[[D80]] [0, 0, 0, 1] : f16 into vector<1x1x2x2xf16>
// CHECK:        %[[D82:.+]] = vector.extract %[[CST_0]][0, 0, 1, 0] : f16 from vector<1x1x2x2xf16>
// CHECK:        %[[D83:.+]] = vector.extract %[[D60]][0, 0, 1, 0] : f16 from vector<1x1x2x2xf16>
// CHECK:        %[[D84:.+]] = vector.insert %[[D83]], %[[CST_2]] [0] : f16 into vector<2xf16>
// CHECK:        %[[D85:.+]] = vector.extract %[[D60]][0, 0, 1, 1] : f16 from vector<1x1x2x2xf16>
// CHECK:        %[[D86:.+]] = vector.insert %[[D85]], %[[D84]] [1] : f16 into vector<2xf16>
// CHECK:        %[[D87:.+]] = vector.bitcast %[[D86]] : vector<2xf16> to vector<1xi32>
// CHECK:        %[[D88:.+]] = vector.extract %[[D87]][0] : i32 from vector<1xi32>
// CHECK:        %[[SHUFFLERESULT_5:.+]], %[[VALID_6:.+]] = gpu.shuffle  xor %[[D88]], %[[C1_I32]], %[[C32_I32]] : i32
// CHECK:        %[[D89:.+]] = vector.broadcast %[[SHUFFLERESULT_5]] : i32 to vector<1xi32>
// CHECK:        %[[D90:.+]] = vector.bitcast %[[D89]] : vector<1xi32> to vector<2xf16>
// CHECK:        %[[D91:.+]] = arith.maximumf %[[D90]], %[[D86]] : vector<2xf16>
// CHECK:        %[[D92:.+]] = vector.bitcast %[[D91]] : vector<2xf16> to vector<1xi32>
// CHECK:        %[[D93:.+]] = vector.extract %[[D92]][0] : i32 from vector<1xi32>
// CHECK:        %[[SHUFFLERESULT_7:.+]], %[[VALID_8:.+]] = gpu.shuffle  xor %[[D93]], %[[C2_I32]], %[[C32_I32]] : i32
// CHECK:        %[[D94:.+]] = vector.broadcast %[[SHUFFLERESULT_7]] : i32 to vector<1xi32>
// CHECK:        %[[D95:.+]] = vector.bitcast %[[D94]] : vector<1xi32> to vector<2xf16>
// CHECK:        %[[D96:.+]] = arith.maximumf %[[D95]], %[[D91]] : vector<2xf16>
// CHECK:        %[[D97:.+]] = vector.extract %[[D96]][0] : f16 from vector<2xf16>
// CHECK:        %[[D98:.+]] = vector.extract %[[D96]][1] : f16 from vector<2xf16>
// CHECK:        %[[D99:.+]] = arith.maximumf %[[D97]], %[[D98]] : f16
// CHECK:        %[[D100:.+]] = arith.maximumf %[[D99]], %[[D82]] : f16
// CHECK:        %[[D101:.+]] = vector.insert %[[D100]], %[[D81]] [0, 0, 1, 0] : f16 into vector<1x1x2x2xf16>
// CHECK:        %[[D102:.+]] = vector.insert %[[D100]], %[[D101]] [0, 0, 1, 1] : f16 into vector<1x1x2x2xf16>
// CHECK:        %[[D103:.+]] = vector.extract %[[D102]][0, 0, 0, 0] : f16 from vector<1x1x2x2xf16>
// CHECK:        memref.store %[[D103]], %[[D2]][%[[D8]], %[[D9]]] : memref<16x8xf16>
// CHECK:        %[[D104:.+]] = vector.extract %[[D102]][0, 0, 0, 1] : f16 from vector<1x1x2x2xf16>
// CHECK:        memref.store %[[D104]], %[[D2]][%[[D8]], %[[D14]]] : memref<16x8xf16>
// CHECK:        %[[D105:.+]] = vector.extract %[[D102]][0, 0, 1, 0] : f16 from vector<1x1x2x2xf16>
// CHECK:        memref.store %[[D105]], %[[D2]][%[[D29]], %[[D9]]] : memref<16x8xf16>
// CHECK:        %[[D106:.+]] = vector.extract %[[D102]][0, 0, 1, 1] : f16 from vector<1x1x2x2xf16>
// CHECK:        memref.store %[[D106]], %[[D2]][%[[D29]], %[[D14]]] : memref<16x8xf16>
// CHECK:        return
// CHECK:      }

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<()[s0] -> (s0 * 16)>
#map2 = affine_map<(d0)[s0] -> (d0 + s0)>
#map3 = affine_map<(d0) -> (d0 * 16)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map5 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map6 = affine_map<(d0, d1, d2) -> (d0, d1)>
builtin.module attributes { transform.with_named_sequence } {
  func.func @matmul_scf() {
    %cst = arith.constant 0.000000e+00 : f16
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : memref<16x64xf16>
    memref.assume_alignment %0, 64 : memref<16x64xf16>
    %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : memref<8x64xf16>
    memref.assume_alignment %1, 64 : memref<8x64xf16>
    %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : memref<16x8xf16>
    memref.assume_alignment %2, 64 : memref<16x8xf16>
    %3 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) alignment(64) offset(%c0) : memref<16x8xf16>
    memref.assume_alignment %3, 64 : memref<16x8xf16>
    %workgroup_id_x = hal.interface.workgroup.id[0] : index
    %4 = affine.apply #map1()[%workgroup_id_x]
    %5 = affine.apply #map2(%c0)[%4]
    %6 = vector.transfer_read %2[%5, %c0], %cst {in_bounds = [true, true]} : memref<16x8xf16>, vector<16x8xf16>
    %7 = scf.for %arg0 = %c0 to %c4 step %c1 iter_args(%arg1 = %6) -> (vector<16x8xf16>) {
      %9 = affine.apply #map3(%arg0)
      %10 = affine.apply #map2(%c0)[%9]
      %11 = vector.transfer_read %0[%c0, %10], %cst {in_bounds = [true, true]} : memref<16x64xf16>, vector<16x16xf16>
      %13 = vector.transfer_read %1[%c0, %10], %cst {in_bounds = [true, true]} : memref<8x64xf16>, vector<8x16xf16>
      %14 = vector.contract {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %11, %13, %arg1 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
      scf.yield %14 : vector<16x8xf16>
    }
    %8 = affine.apply #map2(%c0)[%4]
    vector.transfer_write %7, %3[%8, %c0] {in_bounds = [true, true]} : vector<16x8xf16>, memref<16x8xf16>
    return
  }
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    %transformed_func = transform.iree.layout_analysis_and_distribution %top_level_func : (!transform.any_op) -> (!transform.any_op)
    transform.yield
  }
} // module


// CHECK-DAG:  #[[MAP:.+]] = affine_map<()[s0] -> (s0 * 16)>
// CHECK-DAG:  #[[MAP1:.+]] = affine_map<(d0)[s0] -> (d0 + s0)>
// CHECK-DAG:  #[[MAP2:.+]] = affine_map<(d0, d1, d2) -> (d1 + d2 * 16)>
// CHECK-DAG:  #[[MAP3:.+]] = affine_map<(d0, d1, d2) -> (d0 * 2)>
// CHECK-DAG:  #[[MAP4:.+]] = affine_map<(d0, d1, d2) -> (d0 * 2 + 1)>
// CHECK-DAG:  #[[MAP5:.+]] = affine_map<(d0, d1, d2) -> (d1 + d2 * 16 + 8)>
// CHECK-DAG:  #[[MAP6:.+]] = affine_map<(d0) -> (d0 * 16)>
// CHECK-DAG:  #[[MAP7:.+]] = affine_map<(d0, d1, d2) -> (d0 * 2 + 8)>
// CHECK-DAG:  #[[MAP8:.+]] = affine_map<(d0, d1, d2) -> (d0 * 2 + 9)>
// CHECK-DAG:  #[[MAP9:.+]] = affine_map<(d0, d1, d2) -> (d1 + d2 * 8)>
// CHECK:      func.func @matmul_scf() {
// CHECK-DAG:    %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:    %[[C4:.+]] = arith.constant 4 : index
// CHECK-DAG:    %[[C1:.+]] = arith.constant 1 : index
// CHECK:        %[[D0:.+]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64)
// CHECK-SAME:     offset(%[[C0]]) flags(ReadOnly) : memref<16x64xf16>
// CHECK:        memref.assume_alignment %[[D0]], 64 : memref<16x64xf16>
// CHECK:        %[[D1:.+]] = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64)
// CHECK-SAME:     offset(%[[C0]]) flags(ReadOnly) : memref<8x64xf16>
// CHECK:        memref.assume_alignment %[[D1]], 64 : memref<8x64xf16>
// CHECK:        %[[D2:.+]] = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64)
// CHECK-SAME:     offset(%[[C0]]) flags(ReadOnly) : memref<16x8xf16>
// CHECK:        memref.assume_alignment %[[D2]], 64 : memref<16x8xf16>
// CHECK:        %[[D3:.+]] = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) alignment(64)
// CHECK-SAME:     offset(%[[C0]]) : memref<16x8xf16>
// CHECK:        memref.assume_alignment %[[D3]], 64 : memref<16x8xf16>
// CHECK:        %[[WORKGROUP_ID_X:.+]] = hal.interface.workgroup.id[0] : index
// CHECK-DAG:    %[[D4:.+]] = affine.apply #[[MAP]]()[%[[WORKGROUP_ID_X]]]
// CHECK-DAG:    %[[D5:.+]] = affine.apply #[[MAP1]](%[[C0]])[%[[D4]]]
// CHECK-DAG:    %[[D6:.+]] = gpu.thread_id  x
// CHECK-DAG:    %[[D7:.+]] = gpu.thread_id  y
// CHECK-DAG:    %[[D8:.+]] = gpu.thread_id  z
// CHECK-DAG:    %[[CST:.+]] = arith.constant dense<0.000000e+00> : vector<1x1x2x2xf16>
// CHECK-DAG:    %[[D9:.+]] = affine.apply #[[MAP2]](%[[D6]], %[[D7]], %[[D8]])
// CHECK-DAG:    %[[D10:.+]] = affine.apply #[[MAP3]](%[[D6]], %[[D7]], %[[D8]])
// CHECK:        %[[D11:.+]] = arith.addi %[[D9]], %[[D5]] : index
// CHECK:        %[[D12:.+]] = arith.addi %[[D10]], %[[C0]] : index
// CHECK:        %[[D13:.+]] = memref.load %[[D2]][%[[D11]], %[[D12]]] : memref<16x8xf16>
// CHECK:        %[[D14:.+]] = vector.broadcast %[[D13]] : f16 to vector<1xf16>
// CHECK:        %[[D15:.+]] = vector.insert_strided_slice %[[D14]], %[[CST]] {offsets = [0, 0, 0, 0], strides = [1]} :
// CHECK-SAME:     vector<1xf16> into vector<1x1x2x2xf16>
// CHECK-DAG:    %[[D16:.+]] = affine.apply #[[MAP4]](%[[D6]], %[[D7]], %[[D8]])
// CHECK:        %[[D17:.+]] = arith.addi %[[D16]], %[[C0]] : index
// CHECK:        %[[D18:.+]] = memref.load %[[D2]][%[[D11]], %[[D17]]] : memref<16x8xf16>
// CHECK:        %[[D19:.+]] = vector.broadcast %[[D18]] : f16 to vector<1xf16>
// CHECK:        %[[D20:.+]] = vector.insert_strided_slice %[[D19]], %[[D15]] {offsets = [0, 0, 0, 1], strides = [1]} :
// CHECK-SAME:     vector<1xf16> into vector<1x1x2x2xf16>
// CHECK-DAG:    %[[D21:.+]] = affine.apply #[[MAP5]](%[[D6]], %[[D7]], %[[D8]])
// CHECK:        %[[D22:.+]] = arith.addi %[[D21]], %[[D5]] : index
// CHECK:        %[[D23:.+]] = memref.load %[[D2]][%[[D22]], %[[D12]]] : memref<16x8xf16>
// CHECK:        %[[D24:.+]] = vector.broadcast %[[D23]] : f16 to vector<1xf16>
// CHECK:        %[[D25:.+]] = vector.insert_strided_slice %[[D24]], %[[D20]] {offsets = [0, 0, 1, 0], strides = [1]} :
// CHECK-SAME:     vector<1xf16> into vector<1x1x2x2xf16>
// CHECK:        %[[D26:.+]] = memref.load %[[D2]][%[[D22]], %[[D17]]] : memref<16x8xf16>
// CHECK:        %[[D27:.+]] = vector.broadcast %[[D26]] : f16 to vector<1xf16>
// CHECK:        %[[D28:.+]] = vector.insert_strided_slice %[[D27]], %[[D25]] {offsets = [0, 0, 1, 1], strides = [1]} :
// CHECK-SAME:     vector<1xf16> into vector<1x1x2x2xf16>
// CHECK-DAG:    %[[CST_0:.+]] = arith.constant dense<0.000000e+00> : vector<16x8xf16>
// CHECK:        %[[D29:.+]]:2 = scf.for %[[ARG0:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C4]] step %[[C1]]
// CHECK-SAME:     iter_args(%[[ARG1:[a-zA-Z0-9_]+]] = %[[CST_0]], %[[ARG2:[a-zA-Z0-9_]+]] = %[[D28]]) ->
// CHECK-SAME:     (vector<16x8xf16>, vector<1x1x2x2xf16>) {
// CHECK-DAG:      %[[D34:.+]] = affine.apply #[[MAP6]](%[[ARG0]])
// CHECK-DAG:      %[[D35:.+]] = affine.apply #[[MAP1]](%[[C0]])[%[[D34]]]
// CHECK-DAG:      %[[CST_1:.+]] = arith.constant dense<0.000000e+00> : vector<1x1x4x2xf16>
// CHECK:          %[[D36:.+]] = arith.addi %[[D9]], %[[C0]] : index
// CHECK:          %[[D37:.+]] = arith.addi %[[D10]], %[[D35]] : index
// CHECK:          %[[D38:.+]] = memref.load %[[D0]][%[[D36]], %[[D37]]] : memref<16x64xf16>
// CHECK:          %[[D39:.+]] = vector.broadcast %[[D38]] : f16 to vector<1xf16>
// CHECK:          %[[D40:.+]] = vector.insert_strided_slice %[[D39]], %[[CST_1]] {offsets = [0, 0, 0, 0], strides =
// CHECK-SAME:       [1]} : vector<1xf16> into vector<1x1x4x2xf16>
// CHECK:          %[[D41:.+]] = arith.addi %[[D16]], %[[D35]] : index
// CHECK:          %[[D42:.+]] = memref.load %[[D0]][%[[D36]], %[[D41]]] : memref<16x64xf16>
// CHECK:          %[[D43:.+]] = vector.broadcast %[[D42]] : f16 to vector<1xf16>
// CHECK:          %[[D44:.+]] = vector.insert_strided_slice %[[D43]], %[[D40]] {offsets = [0, 0, 0, 1], strides = [1]}
// CHECK-SAME:       : vector<1xf16> into vector<1x1x4x2xf16>
// CHECK-DAG:      %[[D45:.+]] = affine.apply #[[MAP7]](%[[D6]], %[[D7]], %[[D8]])
// CHECK:          %[[D46:.+]] = arith.addi %[[D45]], %[[D35]] : index
// CHECK:          %[[D47:.+]] = memref.load %[[D0]][%[[D36]], %[[D46]]] : memref<16x64xf16>
// CHECK:          %[[D48:.+]] = vector.broadcast %[[D47]] : f16 to vector<1xf16>
// CHECK:          %[[D49:.+]] = vector.insert_strided_slice %[[D48]], %[[D44]] {offsets = [0, 0, 2, 0], strides = [1]}
// CHECK-SAME:       : vector<1xf16> into vector<1x1x4x2xf16>
// CHECK-DAG:      %[[D50:.+]] = affine.apply #[[MAP8]](%[[D6]], %[[D7]], %[[D8]])
// CHECK:          %[[D51:.+]] = arith.addi %[[D50]], %[[D35]] : index
// CHECK:          %[[D52:.+]] = memref.load %[[D0]][%[[D36]], %[[D51]]] : memref<16x64xf16>
// CHECK:          %[[D53:.+]] = vector.broadcast %[[D52]] : f16 to vector<1xf16>
// CHECK:          %[[D54:.+]] = vector.insert_strided_slice %[[D53]], %[[D49]] {offsets = [0, 0, 2, 1], strides = [1]}
// CHECK-SAME:       : vector<1xf16> into vector<1x1x4x2xf16>
// CHECK:          %[[D55:.+]] = arith.addi %[[D21]], %[[C0]] : index
// CHECK:          %[[D56:.+]] = memref.load %[[D0]][%[[D55]], %[[D37]]] : memref<16x64xf16>
// CHECK:          %[[D57:.+]] = vector.broadcast %[[D56]] : f16 to vector<1xf16>
// CHECK:          %[[D58:.+]] = vector.insert_strided_slice %[[D57]], %[[D54]] {offsets = [0, 0, 1, 0], strides = [1]}
// CHECK-SAME:       : vector<1xf16> into vector<1x1x4x2xf16>
// CHECK:          %[[D59:.+]] = memref.load %[[D0]][%[[D55]], %[[D41]]] : memref<16x64xf16>
// CHECK:          %[[D60:.+]] = vector.broadcast %[[D59]] : f16 to vector<1xf16>
// CHECK:          %[[D61:.+]] = vector.insert_strided_slice %[[D60]], %[[D58]] {offsets = [0, 0, 1, 1], strides = [1]}
// CHECK-SAME:       : vector<1xf16> into vector<1x1x4x2xf16>
// CHECK:          %[[D62:.+]] = memref.load %[[D0]][%[[D55]], %[[D46]]] : memref<16x64xf16>
// CHECK:          %[[D63:.+]] = vector.broadcast %[[D62]] : f16 to vector<1xf16>
// CHECK:          %[[D64:.+]] = vector.insert_strided_slice %[[D63]], %[[D61]] {offsets = [0, 0, 3, 0], strides = [1]}
// CHECK-SAME:       : vector<1xf16> into vector<1x1x4x2xf16>
// CHECK:          %[[D65:.+]] = memref.load %[[D0]][%[[D55]], %[[D51]]] : memref<16x64xf16>
// CHECK:          %[[D66:.+]] = vector.broadcast %[[D65]] : f16 to vector<1xf16>
// CHECK:          %[[D67:.+]] = vector.insert_strided_slice %[[D66]], %[[D64]] {offsets = [0, 0, 3, 1], strides = [1]}
// CHECK-SAME:       : vector<1xf16> into vector<1x1x4x2xf16>
// CHECK-DAG:      %[[D68:.+]] = affine.apply #[[MAP9]](%[[D6]], %[[D7]], %[[D8]])
// CHECK:          %[[D69:.+]] = arith.addi %[[D68]], %[[C0]] : index
// CHECK:          %[[D70:.+]] = memref.load %[[D1]][%[[D69]], %[[D37]]] : memref<8x64xf16>
// CHECK:          %[[D71:.+]] = vector.broadcast %[[D70]] : f16 to vector<1xf16>
// CHECK:          %[[D72:.+]] = vector.insert_strided_slice %[[D71]], %[[CST]] {offsets = [0, 0, 0, 0], strides = [1]}
// CHECK-SAME:       : vector<1xf16> into vector<1x1x2x2xf16>
// CHECK:          %[[D73:.+]] = memref.load %[[D1]][%[[D69]], %[[D41]]] : memref<8x64xf16>
// CHECK:          %[[D74:.+]] = vector.broadcast %[[D73]] : f16 to vector<1xf16>
// CHECK:          %[[D75:.+]] = vector.insert_strided_slice %[[D74]], %[[D72]] {offsets = [0, 0, 0, 1], strides = [1]}
// CHECK-SAME:       : vector<1xf16> into vector<1x1x2x2xf16>
// CHECK:          %[[D76:.+]] = memref.load %[[D1]][%[[D69]], %[[D46]]] : memref<8x64xf16>
// CHECK:          %[[D77:.+]] = vector.broadcast %[[D76]] : f16 to vector<1xf16>
// CHECK:          %[[D78:.+]] = vector.insert_strided_slice %[[D77]], %[[D75]] {offsets = [0, 0, 1, 0], strides = [1]}
// CHECK-SAME:       : vector<1xf16> into vector<1x1x2x2xf16>
// CHECK:          %[[D79:.+]] = memref.load %[[D1]][%[[D69]], %[[D51]]] : memref<8x64xf16>
// CHECK:          %[[D80:.+]] = vector.broadcast %[[D79]] : f16 to vector<1xf16>
// CHECK:          %[[D81:.+]] = vector.insert_strided_slice %[[D80]], %[[D78]] {offsets = [0, 0, 1, 1], strides = [1]}
// CHECK-SAME:       : vector<1xf16> into vector<1x1x2x2xf16>
// CHECK:          %[[D82:.+]] = vector.extract %[[ARG2]][0, 0] : vector<2x2xf16> from vector<1x1x2x2xf16>
// CHECK:          %[[D83:.+]] = vector.extract %[[D67]][0, 0] : vector<4x2xf16> from vector<1x1x4x2xf16>
// CHECK:          %[[D84:.+]] = vector.extract %[[D81]][0, 0] : vector<2x2xf16> from vector<1x1x2x2xf16>
// CHECK:          %[[D85:.+]] = nvgpu.mma.sync(%[[D83]], %[[D84]], %[[D82]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf16>) -> vector<2x2xf16>
// CHECK:          %[[D86:.+]] = vector.insert %[[D85]], %[[CST]] [0, 0] : vector<2x2xf16> into vector<1x1x2x2xf16>
// CHECK:          scf.yield %[[CST_0]], %[[D86]] : vector<16x8xf16>, vector<1x1x2x2xf16>
// CHECK:        }
// CHECK:        %[[D30:.+]] = vector.extract %[[D29]]#[[D1:.+]][0, 0, 0, 0] : f16 from vector<1x1x2x2xf16>
// CHECK:        memref.store %[[D30]], %[[D3]][%[[D11]], %[[D12]]] : memref<16x8xf16>
// CHECK:        %[[D31:.+]] = vector.extract %[[D29]]#[[D1]][0, 0, 0, 1] : f16 from vector<1x1x2x2xf16>
// CHECK:        memref.store %[[D31]], %[[D3]][%[[D11]], %[[D17]]] : memref<16x8xf16>
// CHECK:        %[[D32:.+]] = vector.extract %[[D29]]#[[D1]][0, 0, 1, 0] : f16 from vector<1x1x2x2xf16>
// CHECK:        memref.store %[[D32]], %[[D3]][%[[D22]], %[[D12]]] : memref<16x8xf16>
// CHECK:        %[[D33:.+]] = vector.extract %[[D29]]#[[D1]][0, 0, 1, 1] : f16 from vector<1x1x2x2xf16>
// CHECK:        memref.store %[[D33]], %[[D3]][%[[D22]], %[[D17]]] : memref<16x8xf16>
// CHECK:        return
// CHECK:      }

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<()[s0] -> (s0 * 16)>
#map2 = affine_map<(d0)[s0] -> (d0 + s0)>
#map3 = affine_map<(d0) -> (d0 * 16)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map5 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map6 = affine_map<(d0, d1, d2) -> (d0, d1)>
builtin.module attributes { transform.with_named_sequence } {
  func.func @matmul_scf() {
    %cst = arith.constant 0.000000e+00 : f16
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : memref<16x64xf16>
    memref.assume_alignment %0, 64 : memref<16x64xf16>
    %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : memref<8x64xf16>
    memref.assume_alignment %1, 64 : memref<8x64xf16>
    %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : memref<16x8xf16>
    memref.assume_alignment %2, 64 : memref<16x8xf16>
    %3 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) alignment(64) offset(%c0) : memref<16x8xf16>
    memref.assume_alignment %3, 64 : memref<16x8xf16>
    %workgroup_id_x = hal.interface.workgroup.id[0] : index
    %4 = affine.apply #map1()[%workgroup_id_x]
    %5 = affine.apply #map2(%c0)[%4]
    %cst_2 = arith.constant dense<0.000000e+00> : vector<16x8xf16>
    %7 = scf.for %arg0 = %c0 to %c4 step %c1 iter_args(%arg1 = %cst_2) -> (vector<16x8xf16>) {
      %9 = affine.apply #map3(%arg0)
      %10 = affine.apply #map2(%c0)[%9]
      %11 = vector.transfer_read %0[%c0, %10], %cst {in_bounds = [true, true]} : memref<16x64xf16>, vector<16x16xf16>
      %13 = vector.transfer_read %1[%c0, %10], %cst {in_bounds = [true, true]} : memref<8x64xf16>, vector<8x16xf16>
      %14 = vector.contract {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %11, %13, %arg1 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
      scf.yield %14 : vector<16x8xf16>
    }
    %8 = affine.apply #map2(%c0)[%4]
    vector.transfer_write %7, %3[%8, %c0] {in_bounds = [true, true]} : vector<16x8xf16>, memref<16x8xf16>
    return
  }
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    %transformed_func = transform.iree.layout_analysis_and_distribution %top_level_func : (!transform.any_op) -> (!transform.any_op)
    transform.yield
  }
} // module


// CHECK-DAG:  #[[MAP:.+]] = affine_map<()[s0] -> (s0 * 16)>
// CHECK-DAG:  #[[MAP1:.+]] = affine_map<(d0) -> (d0 * 16)>
// CHECK-DAG:  #[[MAP2:.+]] = affine_map<(d0)[s0] -> (d0 + s0)>
// CHECK-DAG:  #[[MAP3:.+]] = affine_map<(d0, d1, d2) -> (d1 + d2 * 16)>
// CHECK-DAG:  #[[MAP4:.+]] = affine_map<(d0, d1, d2) -> (d0 * 2)>
// CHECK-DAG:  #[[MAP5:.+]] = affine_map<(d0, d1, d2) -> (d0 * 2 + 1)>
// CHECK-DAG:  #[[MAP6:.+]] = affine_map<(d0, d1, d2) -> (d0 * 2 + 8)>
// CHECK-DAG:  #[[MAP7:.+]] = affine_map<(d0, d1, d2) -> (d0 * 2 + 9)>
// CHECK-DAG:  #[[MAP8:.+]] = affine_map<(d0, d1, d2) -> (d1 + d2 * 16 + 8)>
// CHECK-DAG:  #[[MAP9:.+]] = affine_map<(d0, d1, d2) -> (d1 + d2 * 8)>
// CHECK:      func.func @matmul_scf() {
// CHECK-DAG:    %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:    %[[C4:.+]] = arith.constant 4 : index
// CHECK-DAG:    %[[C1:.+]] = arith.constant 1 : index
// CHECK:        %[[D0:.+]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64)
// CHECK-SAME:     offset(%[[C0]]) flags(ReadOnly) : memref<16x64xf16>
// CHECK:        memref.assume_alignment %[[D0]], 64 : memref<16x64xf16>
// CHECK:        %[[D1:.+]] = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64)
// CHECK-SAME:     offset(%[[C0]]) flags(ReadOnly) : memref<8x64xf16>
// CHECK:        memref.assume_alignment %[[D1]], 64 : memref<8x64xf16>
// CHECK:        %[[D2:.+]] = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64)
// CHECK-SAME:     offset(%[[C0]]) flags(ReadOnly) : memref<16x8xf16>
// CHECK:        memref.assume_alignment %[[D2]], 64 : memref<16x8xf16>
// CHECK:        %[[D3:.+]] = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) alignment(64)
// CHECK-SAME:     offset(%[[C0]]) : memref<16x8xf16>
// CHECK:        memref.assume_alignment %[[D3]], 64 : memref<16x8xf16>
// CHECK:        %[[WORKGROUP_ID_X:.+]] = hal.interface.workgroup.id[0] : index
// CHECK-DAG:    %[[D4:.+]] = affine.apply #[[MAP]]()[%[[WORKGROUP_ID_X]]]
// CHECK-DAG:    %[[CST:.+]] = arith.constant dense<0.000000e+00> : vector<1x1x2x2xf16>
// CHECK-DAG:    %[[CST_0:.+]] = arith.constant dense<0.000000e+00> : vector<16x8xf16>
// CHECK:        %[[D5:.+]]:2 = scf.for %[[ARG0:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C4]] step %[[C1]]
// CHECK-SAME:     iter_args(%[[ARG1:[a-zA-Z0-9_]+]] = %[[CST_0]], %[[ARG2:[a-zA-Z0-9_]+]] = %[[CST]]) ->
// CHECK-SAME:     (vector<16x8xf16>, vector<1x1x2x2xf16>) {
// CHECK-DAG:      %[[D22:.+]] = affine.apply #[[MAP1]](%[[ARG0]])
// CHECK-DAG:      %[[D23:.+]] = affine.apply #[[MAP2]](%[[C0]])[%[[D22]]]
// CHECK-DAG:      %[[D24:.+]] = gpu.thread_id  x
// CHECK-DAG:      %[[D25:.+]] = gpu.thread_id  y
// CHECK-DAG:      %[[D26:.+]] = gpu.thread_id  z
// CHECK-DAG:      %[[CST_1:.+]] = arith.constant dense<0.000000e+00> : vector<1x1x4x2xf16>
// CHECK-DAG:      %[[D27:.+]] = affine.apply #[[MAP3]](%[[D24]], %[[D25]], %[[D26]])
// CHECK-DAG:      %[[D28:.+]] = affine.apply #[[MAP4]](%[[D24]], %[[D25]], %[[D26]])
// CHECK:          %[[D29:.+]] = arith.addi %[[D27]], %[[C0]] : index
// CHECK:          %[[D30:.+]] = arith.addi %[[D28]], %[[D23]] : index
// CHECK:          %[[D31:.+]] = memref.load %[[D0]][%[[D29]], %[[D30]]] : memref<16x64xf16>
// CHECK:          %[[D32:.+]] = vector.broadcast %[[D31]] : f16 to vector<1xf16>
// CHECK:          %[[D33:.+]] = vector.insert_strided_slice %[[D32]], %[[CST_1]] {offsets = [0, 0, 0, 0], strides =
// CHECK-SAME:       [1]} : vector<1xf16> into vector<1x1x4x2xf16>
// CHECK-DAG:      %[[D34:.+]] = affine.apply #[[MAP5]](%[[D24]], %[[D25]], %[[D26]])
// CHECK:          %[[D35:.+]] = arith.addi %[[D34]], %[[D23]] : index
// CHECK:          %[[D36:.+]] = memref.load %[[D0]][%[[D29]], %[[D35]]] : memref<16x64xf16>
// CHECK:          %[[D37:.+]] = vector.broadcast %[[D36]] : f16 to vector<1xf16>
// CHECK:          %[[D38:.+]] = vector.insert_strided_slice %[[D37]], %[[D33]] {offsets = [0, 0, 0, 1], strides = [1]}
// CHECK-SAME:       : vector<1xf16> into vector<1x1x4x2xf16>
// CHECK-DAG:      %[[D39:.+]] = affine.apply #[[MAP6]](%[[D24]], %[[D25]], %[[D26]])
// CHECK:          %[[D40:.+]] = arith.addi %[[D39]], %[[D23]] : index
// CHECK:          %[[D41:.+]] = memref.load %[[D0]][%[[D29]], %[[D40]]] : memref<16x64xf16>
// CHECK:          %[[D42:.+]] = vector.broadcast %[[D41]] : f16 to vector<1xf16>
// CHECK:          %[[D43:.+]] = vector.insert_strided_slice %[[D42]], %[[D38]] {offsets = [0, 0, 2, 0], strides = [1]}
// CHECK-SAME:       : vector<1xf16> into vector<1x1x4x2xf16>
// CHECK-DAG:      %[[D44:.+]] = affine.apply #[[MAP7]](%[[D24]], %[[D25]], %[[D26]])
// CHECK:          %[[D45:.+]] = arith.addi %[[D44]], %[[D23]] : index
// CHECK:          %[[D46:.+]] = memref.load %[[D0]][%[[D29]], %[[D45]]] : memref<16x64xf16>
// CHECK:          %[[D47:.+]] = vector.broadcast %[[D46]] : f16 to vector<1xf16>
// CHECK:          %[[D48:.+]] = vector.insert_strided_slice %[[D47]], %[[D43]] {offsets = [0, 0, 2, 1], strides = [1]}
// CHECK-SAME:       : vector<1xf16> into vector<1x1x4x2xf16>
// CHECK-DAG:      %[[D49:.+]] = affine.apply #[[MAP8]](%[[D24]], %[[D25]], %[[D26]])
// CHECK:          %[[D50:.+]] = arith.addi %[[D49]], %[[C0]] : index
// CHECK:          %[[D51:.+]] = memref.load %[[D0]][%[[D50]], %[[D30]]] : memref<16x64xf16>
// CHECK:          %[[D52:.+]] = vector.broadcast %[[D51]] : f16 to vector<1xf16>
// CHECK:          %[[D53:.+]] = vector.insert_strided_slice %[[D52]], %[[D48]] {offsets = [0, 0, 1, 0], strides = [1]}
// CHECK-SAME:       : vector<1xf16> into vector<1x1x4x2xf16>
// CHECK:          %[[D54:.+]] = memref.load %[[D0]][%[[D50]], %[[D35]]] : memref<16x64xf16>
// CHECK:          %[[D55:.+]] = vector.broadcast %[[D54]] : f16 to vector<1xf16>
// CHECK:          %[[D56:.+]] = vector.insert_strided_slice %[[D55]], %[[D53]] {offsets = [0, 0, 1, 1], strides = [1]}
// CHECK-SAME:       : vector<1xf16> into vector<1x1x4x2xf16>
// CHECK:          %[[D57:.+]] = memref.load %[[D0]][%[[D50]], %[[D40]]] : memref<16x64xf16>
// CHECK:          %[[D58:.+]] = vector.broadcast %[[D57]] : f16 to vector<1xf16>
// CHECK:          %[[D59:.+]] = vector.insert_strided_slice %[[D58]], %[[D56]] {offsets = [0, 0, 3, 0], strides = [1]}
// CHECK-SAME:       : vector<1xf16> into vector<1x1x4x2xf16>
// CHECK:          %[[D60:.+]] = memref.load %[[D0]][%[[D50]], %[[D45]]] : memref<16x64xf16>
// CHECK:          %[[D61:.+]] = vector.broadcast %[[D60]] : f16 to vector<1xf16>
// CHECK:          %[[D62:.+]] = vector.insert_strided_slice %[[D61]], %[[D59]] {offsets = [0, 0, 3, 1], strides = [1]}
// CHECK-SAME:       : vector<1xf16> into vector<1x1x4x2xf16>
// CHECK-DAG:      %[[D63:.+]] = affine.apply #[[MAP9]](%[[D24]], %[[D25]], %[[D26]])
// CHECK:          %[[D64:.+]] = arith.addi %[[D63]], %[[C0]] : index
// CHECK:          %[[D65:.+]] = memref.load %[[D1]][%[[D64]], %[[D30]]] : memref<8x64xf16>
// CHECK:          %[[D66:.+]] = vector.broadcast %[[D65]] : f16 to vector<1xf16>
// CHECK:          %[[D67:.+]] = vector.insert_strided_slice %[[D66]], %[[CST]] {offsets = [0, 0, 0, 0], strides = [1]}
// CHECK-SAME:       : vector<1xf16> into vector<1x1x2x2xf16>
// CHECK:          %[[D68:.+]] = memref.load %[[D1]][%[[D64]], %[[D35]]] : memref<8x64xf16>
// CHECK:          %[[D69:.+]] = vector.broadcast %[[D68]] : f16 to vector<1xf16>
// CHECK:          %[[D70:.+]] = vector.insert_strided_slice %[[D69]], %[[D67]] {offsets = [0, 0, 0, 1], strides = [1]}
// CHECK-SAME:       : vector<1xf16> into vector<1x1x2x2xf16>
// CHECK:          %[[D71:.+]] = memref.load %[[D1]][%[[D64]], %[[D40]]] : memref<8x64xf16>
// CHECK:          %[[D72:.+]] = vector.broadcast %[[D71]] : f16 to vector<1xf16>
// CHECK:          %[[D73:.+]] = vector.insert_strided_slice %[[D72]], %[[D70]] {offsets = [0, 0, 1, 0], strides = [1]}
// CHECK-SAME:       : vector<1xf16> into vector<1x1x2x2xf16>
// CHECK:          %[[D74:.+]] = memref.load %[[D1]][%[[D64]], %[[D45]]] : memref<8x64xf16>
// CHECK:          %[[D75:.+]] = vector.broadcast %[[D74]] : f16 to vector<1xf16>
// CHECK:          %[[D76:.+]] = vector.insert_strided_slice %[[D75]], %[[D73]] {offsets = [0, 0, 1, 1], strides = [1]}
// CHECK-SAME:       : vector<1xf16> into vector<1x1x2x2xf16>
// CHECK:          %[[D77:.+]] = vector.extract %[[ARG2]][0, 0] : vector<2x2xf16> from vector<1x1x2x2xf16>
// CHECK:          %[[D78:.+]] = vector.extract %[[D62]][0, 0] : vector<4x2xf16> from vector<1x1x4x2xf16>
// CHECK:          %[[D79:.+]] = vector.extract %[[D76]][0, 0] : vector<2x2xf16> from vector<1x1x2x2xf16>
// CHECK:          %[[D80:.+]] = nvgpu.mma.sync(%[[D78]], %[[D79]], %[[D77]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:       (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf16>) -> vector<2x2xf16>
// CHECK:          %[[D81:.+]] = vector.insert %[[D80]], %[[CST]] [0, 0] : vector<2x2xf16> into vector<1x1x2x2xf16>
// CHECK:          scf.yield %[[CST_0]], %[[D81]] : vector<16x8xf16>, vector<1x1x2x2xf16>
// CHECK:        }
// CHECK-DAG:    %[[D6:.+]] = affine.apply #[[MAP2]](%[[C0]])[%[[D4]]]
// CHECK-DAG:    %[[D7:.+]] = gpu.thread_id  x
// CHECK-DAG:    %[[D8:.+]] = gpu.thread_id  y
// CHECK-DAG:    %[[D9:.+]] = gpu.thread_id  z
// CHECK:        %[[D10:.+]] = vector.extract %[[D5]]#[[D1:.+]][0, 0, 0, 0] : f16 from vector<1x1x2x2xf16>
// CHECK-DAG:    %[[D11:.+]] = affine.apply #[[MAP3]](%[[D7]], %[[D8]], %[[D9]])
// CHECK-DAG:    %[[D12:.+]] = affine.apply #[[MAP4]](%[[D7]], %[[D8]], %[[D9]])
// CHECK:        %[[D13:.+]] = arith.addi %[[D11]], %[[D6]] : index
// CHECK:        %[[D14:.+]] = arith.addi %[[D12]], %[[C0]] : index
// CHECK:        memref.store %[[D10]], %[[D3]][%[[D13]], %[[D14]]] : memref<16x8xf16>
// CHECK:        %[[D15:.+]] = vector.extract %[[D5]]#[[D1]][0, 0, 0, 1] : f16 from vector<1x1x2x2xf16>
// CHECK-DAG:    %[[D16:.+]] = affine.apply #[[MAP5]](%[[D7]], %[[D8]], %[[D9]])
// CHECK:        %[[D17:.+]] = arith.addi %[[D16]], %[[C0]] : index
// CHECK:        memref.store %[[D15]], %[[D3]][%[[D13]], %[[D17]]] : memref<16x8xf16>
// CHECK:        %[[D18:.+]] = vector.extract %[[D5]]#[[D1]][0, 0, 1, 0] : f16 from vector<1x1x2x2xf16>
// CHECK-DAG:    %[[D19:.+]] = affine.apply #[[MAP8]](%[[D7]], %[[D8]], %[[D9]])
// CHECK:        %[[D20:.+]] = arith.addi %[[D19]], %[[D6]] : index
// CHECK:        memref.store %[[D18]], %[[D3]][%[[D20]], %[[D14]]] : memref<16x8xf16>
// CHECK:        %[[D21:.+]] = vector.extract %[[D5]]#[[D1]][0, 0, 1, 1] : f16 from vector<1x1x2x2xf16>
// CHECK:        memref.store %[[D21]], %[[D3]][%[[D20]], %[[D17]]] : memref<16x8xf16>
// CHECK:        return
// CHECK:      }

// -----

#map2 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d1)>
builtin.module attributes { transform.with_named_sequence } {
  func.func @matmul_dispatch_0_matmul_16x8x16() {
    %c0 = arith.constant 0 : index
    %cst = arith.constant dense<0.000000e+00> : vector<16x8xf16>
    %cst_1 = arith.constant 0.000000e+00 : f16
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : memref<16x16xf16>
    memref.assume_alignment %0, 64 : memref<16x16xf16>
    %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : memref<8x16xf16>
    memref.assume_alignment %1, 64 : memref<8x16xf16>
    %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : memref<16x8xf16>
    memref.assume_alignment %2, 64 : memref<16x8xf16>
    %3 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) alignment(64) offset(%c0) : memref<16x8xf16>
    memref.assume_alignment %3, 64 : memref<16x8xf16>
    %5 = vector.transfer_read %0[%c0, %c0], %cst_1 {in_bounds = [true, true]} : memref<16x16xf16>, vector<16x16xf16>
    %6 = vector.transfer_read %1[%c0, %c0], %cst_1 {in_bounds = [true, true]} : memref<8x16xf16>, vector<8x16xf16>
    %7 = vector.contract {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %5, %6, %cst : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
    %subview = memref.subview %3[%c0, 0] [16, 8] [1, 1] : memref<16x8xf16> to memref<16x8xf16, strided<[8, 1], offset: ?>>
    %8 = vector.transfer_read %2[%c0, %c0], %cst_1 {in_bounds = [true, true]} : memref<16x8xf16>, vector<16x8xf16>
    %9 = arith.subf %7, %8 : vector<16x8xf16>
    %10 = math.exp %9 : vector<16x8xf16>
    vector.transfer_write %10, %subview[%c0, %c0] {in_bounds = [true, true]} : vector<16x8xf16>, memref<16x8xf16, strided<[8, 1], offset: ?>>
    return
  }
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    %transformed_func = transform.iree.layout_analysis_and_distribution %top_level_func : (!transform.any_op) -> (!transform.any_op)
    transform.yield
  }
} // module


// CHECK-DAG:  #[[MAP:.+]] = affine_map<(d0, d1, d2) -> (d1 + d2 * 16)>
// CHECK-DAG:  #[[MAP1:.+]] = affine_map<(d0, d1, d2) -> (d0 * 2)>
// CHECK-DAG:  #[[MAP2:.+]] = affine_map<(d0, d1, d2) -> (d0 * 2 + 1)>
// CHECK-DAG:  #[[MAP3:.+]] = affine_map<(d0, d1, d2) -> (d0 * 2 + 8)>
// CHECK-DAG:  #[[MAP4:.+]] = affine_map<(d0, d1, d2) -> (d0 * 2 + 9)>
// CHECK-DAG:  #[[MAP5:.+]] = affine_map<(d0, d1, d2) -> (d1 + d2 * 16 + 8)>
// CHECK-DAG:  #[[MAP6:.+]] = affine_map<(d0, d1, d2) -> (d1 + d2 * 8)>
// CHECK:      func.func @matmul_dispatch_0_matmul_16x8x16() {
// CHECK-DAG:    %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:    %[[CST:.+]] = arith.constant dense<0.000000e+00> : vector<1x1x2x2xf16>
// CHECK:        %[[D0:.+]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64)
// CHECK-SAME:     offset(%[[C0]]) flags(ReadOnly) : memref<16x16xf16>
// CHECK:        memref.assume_alignment %[[D0]], 64 : memref<16x16xf16>
// CHECK:        %[[D1:.+]] = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64)
// CHECK-SAME:     offset(%[[C0]]) flags(ReadOnly) : memref<8x16xf16>
// CHECK:        memref.assume_alignment %[[D1]], 64 : memref<8x16xf16>
// CHECK:        %[[D2:.+]] = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64)
// CHECK-SAME:     offset(%[[C0]]) flags(ReadOnly) : memref<16x8xf16>
// CHECK:        memref.assume_alignment %[[D2]], 64 : memref<16x8xf16>
// CHECK:        %[[D3:.+]] = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) alignment(64)
// CHECK-SAME:     offset(%[[C0]]) : memref<16x8xf16>
// CHECK:        memref.assume_alignment %[[D3]], 64 : memref<16x8xf16>
// CHECK-DAG:    %[[D4:.+]] = gpu.thread_id  x
// CHECK-DAG:    %[[D5:.+]] = gpu.thread_id  y
// CHECK-DAG:    %[[D6:.+]] = gpu.thread_id  z
// CHECK-DAG:    %[[CST_0:.+]] = arith.constant dense<0.000000e+00> : vector<1x1x4x2xf16>
// CHECK-DAG:    %[[D7:.+]] = affine.apply #[[MAP]](%[[D4]], %[[D5]], %[[D6]])
// CHECK-DAG:    %[[D8:.+]] = affine.apply #[[MAP1]](%[[D4]], %[[D5]], %[[D6]])
// CHECK:        %[[D9:.+]] = arith.addi %[[D7]], %[[C0]] : index
// CHECK:        %[[D10:.+]] = arith.addi %[[D8]], %[[C0]] : index
// CHECK:        %[[D11:.+]] = memref.load %[[D0]][%[[D9]], %[[D10]]] : memref<16x16xf16>
// CHECK:        %[[D12:.+]] = vector.broadcast %[[D11]] : f16 to vector<1xf16>
// CHECK:        %[[D13:.+]] = vector.insert_strided_slice %[[D12]], %[[CST_0]] {offsets = [0, 0, 0, 0], strides = [1]}
// CHECK-SAME:     : vector<1xf16> into vector<1x1x4x2xf16>
// CHECK-DAG:    %[[D14:.+]] = affine.apply #[[MAP2]](%[[D4]], %[[D5]], %[[D6]])
// CHECK:        %[[D15:.+]] = arith.addi %[[D14]], %[[C0]] : index
// CHECK:        %[[D16:.+]] = memref.load %[[D0]][%[[D9]], %[[D15]]] : memref<16x16xf16>
// CHECK:        %[[D17:.+]] = vector.broadcast %[[D16]] : f16 to vector<1xf16>
// CHECK:        %[[D18:.+]] = vector.insert_strided_slice %[[D17]], %[[D13]] {offsets = [0, 0, 0, 1], strides = [1]} :
// CHECK-SAME:     vector<1xf16> into vector<1x1x4x2xf16>
// CHECK-DAG:    %[[D19:.+]] = affine.apply #[[MAP3]](%[[D4]], %[[D5]], %[[D6]])
// CHECK:        %[[D20:.+]] = arith.addi %[[D19]], %[[C0]] : index
// CHECK:        %[[D21:.+]] = memref.load %[[D0]][%[[D9]], %[[D20]]] : memref<16x16xf16>
// CHECK:        %[[D22:.+]] = vector.broadcast %[[D21]] : f16 to vector<1xf16>
// CHECK:        %[[D23:.+]] = vector.insert_strided_slice %[[D22]], %[[D18]] {offsets = [0, 0, 2, 0], strides = [1]} :
// CHECK-SAME:     vector<1xf16> into vector<1x1x4x2xf16>
// CHECK-DAG:    %[[D24:.+]] = affine.apply #[[MAP4]](%[[D4]], %[[D5]], %[[D6]])
// CHECK:        %[[D25:.+]] = arith.addi %[[D24]], %[[C0]] : index
// CHECK:        %[[D26:.+]] = memref.load %[[D0]][%[[D9]], %[[D25]]] : memref<16x16xf16>
// CHECK:        %[[D27:.+]] = vector.broadcast %[[D26]] : f16 to vector<1xf16>
// CHECK:        %[[D28:.+]] = vector.insert_strided_slice %[[D27]], %[[D23]] {offsets = [0, 0, 2, 1], strides = [1]} :
// CHECK-SAME:     vector<1xf16> into vector<1x1x4x2xf16>
// CHECK-DAG:    %[[D29:.+]] = affine.apply #[[MAP5]](%[[D4]], %[[D5]], %[[D6]])
// CHECK:        %[[D30:.+]] = arith.addi %[[D29]], %[[C0]] : index
// CHECK:        %[[D31:.+]] = memref.load %[[D0]][%[[D30]], %[[D10]]] : memref<16x16xf16>
// CHECK:        %[[D32:.+]] = vector.broadcast %[[D31]] : f16 to vector<1xf16>
// CHECK:        %[[D33:.+]] = vector.insert_strided_slice %[[D32]], %[[D28]] {offsets = [0, 0, 1, 0], strides = [1]} :
// CHECK-SAME:     vector<1xf16> into vector<1x1x4x2xf16>
// CHECK:        %[[D34:.+]] = memref.load %[[D0]][%[[D30]], %[[D15]]] : memref<16x16xf16>
// CHECK:        %[[D35:.+]] = vector.broadcast %[[D34]] : f16 to vector<1xf16>
// CHECK:        %[[D36:.+]] = vector.insert_strided_slice %[[D35]], %[[D33]] {offsets = [0, 0, 1, 1], strides = [1]} :
// CHECK-SAME:     vector<1xf16> into vector<1x1x4x2xf16>
// CHECK:        %[[D37:.+]] = memref.load %[[D0]][%[[D30]], %[[D20]]] : memref<16x16xf16>
// CHECK:        %[[D38:.+]] = vector.broadcast %[[D37]] : f16 to vector<1xf16>
// CHECK:        %[[D39:.+]] = vector.insert_strided_slice %[[D38]], %[[D36]] {offsets = [0, 0, 3, 0], strides = [1]} :
// CHECK-SAME:     vector<1xf16> into vector<1x1x4x2xf16>
// CHECK:        %[[D40:.+]] = memref.load %[[D0]][%[[D30]], %[[D25]]] : memref<16x16xf16>
// CHECK:        %[[D41:.+]] = vector.broadcast %[[D40]] : f16 to vector<1xf16>
// CHECK:        %[[D42:.+]] = vector.insert_strided_slice %[[D41]], %[[D39]] {offsets = [0, 0, 3, 1], strides = [1]} :
// CHECK-SAME:     vector<1xf16> into vector<1x1x4x2xf16>
// CHECK-DAG:    %[[D43:.+]] = affine.apply #[[MAP6]](%[[D4]], %[[D5]], %[[D6]])
// CHECK:        %[[D44:.+]] = arith.addi %[[D43]], %[[C0]] : index
// CHECK:        %[[D45:.+]] = memref.load %[[D1]][%[[D44]], %[[D10]]] : memref<8x16xf16>
// CHECK:        %[[D46:.+]] = vector.broadcast %[[D45]] : f16 to vector<1xf16>
// CHECK:        %[[D47:.+]] = vector.insert_strided_slice %[[D46]], %[[CST]] {offsets = [0, 0, 0, 0], strides = [1]} :
// CHECK-SAME:     vector<1xf16> into vector<1x1x2x2xf16>
// CHECK:        %[[D48:.+]] = memref.load %[[D1]][%[[D44]], %[[D15]]] : memref<8x16xf16>
// CHECK:        %[[D49:.+]] = vector.broadcast %[[D48]] : f16 to vector<1xf16>
// CHECK:        %[[D50:.+]] = vector.insert_strided_slice %[[D49]], %[[D47]] {offsets = [0, 0, 0, 1], strides = [1]} :
// CHECK-SAME:     vector<1xf16> into vector<1x1x2x2xf16>
// CHECK:        %[[D51:.+]] = memref.load %[[D1]][%[[D44]], %[[D20]]] : memref<8x16xf16>
// CHECK:        %[[D52:.+]] = vector.broadcast %[[D51]] : f16 to vector<1xf16>
// CHECK:        %[[D53:.+]] = vector.insert_strided_slice %[[D52]], %[[D50]] {offsets = [0, 0, 1, 0], strides = [1]} :
// CHECK-SAME:     vector<1xf16> into vector<1x1x2x2xf16>
// CHECK:        %[[D54:.+]] = memref.load %[[D1]][%[[D44]], %[[D25]]] : memref<8x16xf16>
// CHECK:        %[[D55:.+]] = vector.broadcast %[[D54]] : f16 to vector<1xf16>
// CHECK:        %[[D56:.+]] = vector.insert_strided_slice %[[D55]], %[[D53]] {offsets = [0, 0, 1, 1], strides = [1]} :
// CHECK-SAME:     vector<1xf16> into vector<1x1x2x2xf16>
// CHECK:        %[[D57:.+]] = vector.extract %[[CST]][0, 0] : vector<2x2xf16> from vector<1x1x2x2xf16>
// CHECK:        %[[D58:.+]] = vector.extract %[[D42]][0, 0] : vector<4x2xf16> from vector<1x1x4x2xf16>
// CHECK:        %[[D59:.+]] = vector.extract %[[D56]][0, 0] : vector<2x2xf16> from vector<1x1x2x2xf16>
// CHECK:        %[[D60:.+]] = nvgpu.mma.sync(%[[D58]], %[[D59]], %[[D57]]) {mmaShape = [16, 8, 16]} : (vector<4x2xf16>,
// CHECK-SAME:     vector<2x2xf16>, vector<2x2xf16>) -> vector<2x2xf16>
// CHECK:        %[[D61:.+]] = vector.insert %[[D60]], %[[CST]] [0, 0] : vector<2x2xf16> into vector<1x1x2x2xf16>
// CHECK:        %[[SUBVIEW:.+]] = memref.subview %[[D3]][%[[C0]], 0] [16, 8] [1, 1] : memref<16x8xf16> to
// CHECK-SAME:     memref<16x8xf16, strided<[8, 1], offset: ?>>
// CHECK:        %[[D62:.+]] = memref.load %[[D2]][%[[D9]], %[[D10]]] : memref<16x8xf16>
// CHECK:        %[[D63:.+]] = vector.broadcast %[[D62]] : f16 to vector<1xf16>
// CHECK:        %[[D64:.+]] = vector.insert_strided_slice %[[D63]], %[[CST]] {offsets = [0, 0, 0, 0], strides = [1]} :
// CHECK-SAME:     vector<1xf16> into vector<1x1x2x2xf16>
// CHECK:        %[[D65:.+]] = memref.load %[[D2]][%[[D9]], %[[D15]]] : memref<16x8xf16>
// CHECK:        %[[D66:.+]] = vector.broadcast %[[D65]] : f16 to vector<1xf16>
// CHECK:        %[[D67:.+]] = vector.insert_strided_slice %[[D66]], %[[D64]] {offsets = [0, 0, 0, 1], strides = [1]} :
// CHECK-SAME:     vector<1xf16> into vector<1x1x2x2xf16>
// CHECK:        %[[D68:.+]] = memref.load %[[D2]][%[[D30]], %[[D10]]] : memref<16x8xf16>
// CHECK:        %[[D69:.+]] = vector.broadcast %[[D68]] : f16 to vector<1xf16>
// CHECK:        %[[D70:.+]] = vector.insert_strided_slice %[[D69]], %[[D67]] {offsets = [0, 0, 1, 0], strides = [1]} :
// CHECK-SAME:     vector<1xf16> into vector<1x1x2x2xf16>
// CHECK:        %[[D71:.+]] = memref.load %[[D2]][%[[D30]], %[[D15]]] : memref<16x8xf16>
// CHECK:        %[[D72:.+]] = vector.broadcast %[[D71]] : f16 to vector<1xf16>
// CHECK:        %[[D73:.+]] = vector.insert_strided_slice %[[D72]], %[[D70]] {offsets = [0, 0, 1, 1], strides = [1]} :
// CHECK-SAME:     vector<1xf16> into vector<1x1x2x2xf16>
// CHECK:        %[[D74:.+]] = arith.subf %[[D61]], %[[D73]] : vector<1x1x2x2xf16>
// CHECK:        %[[D75:.+]] = math.exp %[[D74]] : vector<1x1x2x2xf16>
// CHECK:        %[[D76:.+]] = vector.extract %[[D75]][0, 0, 0, 0] : f16 from vector<1x1x2x2xf16>
// CHECK:        memref.store %[[D76]], %[[SUBVIEW]][%[[D9]], %[[D10]]] : memref<16x8xf16, strided<[8, 1], offset: ?>>
// CHECK:        %[[D77:.+]] = vector.extract %[[D75]][0, 0, 0, 1] : f16 from vector<1x1x2x2xf16>
// CHECK:        memref.store %[[D77]], %[[SUBVIEW]][%[[D9]], %[[D15]]] : memref<16x8xf16, strided<[8, 1], offset: ?>>
// CHECK:        %[[D78:.+]] = vector.extract %[[D75]][0, 0, 1, 0] : f16 from vector<1x1x2x2xf16>
// CHECK:        memref.store %[[D78]], %[[SUBVIEW]][%[[D30]], %[[D10]]] : memref<16x8xf16, strided<[8, 1], offset: ?>>
// CHECK:        %[[D79:.+]] = vector.extract %[[D75]][0, 0, 1, 1] : f16 from vector<1x1x2x2xf16>
// CHECK:        memref.store %[[D79]], %[[SUBVIEW]][%[[D30]], %[[D15]]] : memref<16x8xf16, strided<[8, 1], offset: ?>>
// CHECK:        return
// CHECK:      }

// -----

builtin.module attributes { transform.with_named_sequence } {
  func.func @matmul_dispatch_0_matmul_16x8x16_shared() {
    %c0 = arith.constant 0 : index
    %cst = arith.constant dense<0.000000e+00> : vector<16x8xf16>
    %cst_0 = arith.constant 0.000000e+00 : f16
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : memref<16x16xf16, #gpu.address_space<workgroup>>
    %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : memref<8x16xf16, #gpu.address_space<workgroup>>
    %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : memref<16x8xf16>
    memref.assume_alignment %2, 64 : memref<16x8xf16>
    %3 = vector.transfer_read %0[%c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<16x16xf16, #gpu.address_space<workgroup>>, vector<16x16xf16>
    %4 = vector.transfer_read %1[%c0, %c0], %cst_0 {permutation_map = affine_map<(d0, d1) -> (d1, d0)>, in_bounds = [true, true]} : memref<8x16xf16, #gpu.address_space<workgroup>>, vector<8x16xf16>
    %5 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %3, %4, %cst : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
    vector.transfer_write %5, %2[%c0, %c0] {in_bounds = [true, true]} : vector<16x8xf16>, memref<16x8xf16>
    return
  }
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    %transformed_func = transform.iree.layout_analysis_and_distribution %top_level_func : (!transform.any_op) -> (!transform.any_op)
    transform.yield
  }
} // module

// CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2) -> (d1 + d2 * 16)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2) -> (d0 * 2)>
// CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1) -> ((d0 + d1 * 4) mod 8)>
// CHECK-DAG: #[[MAP3:.+]] = affine_map<(d0, d1, d2) -> (d0 * 2 + 8)>
// CHECK-DAG: #[[MAP4:.+]] = affine_map<(d0, d1, d2) -> (d1 + d2 * 16 + 8)>
// CHECK-DAG: #[[MAP5:.+]] = affine_map<(d0, d1, d2) -> (d1 + d2 * 8)>
// CHECK-DAG: #[[MAP6:.+]] = affine_map<(d0, d1, d2) -> (d0 * 2 + 1)>
    // CHECK: func.func @matmul_dispatch_0_matmul_16x8x16_shared() {
// CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:   %[[TX:.+]] = gpu.thread_id  x
// CHECK-DAG:   %[[TY:.+]] = gpu.thread_id  y
// CHECK-DAG:   %[[VECOFF0:.+]] = affine.apply #[[MAP0]](%[[C0]], %[[C0]], %[[C0]])
// CHECK-DAG:   %[[VECOFF1:.+]] = affine.apply #[[MAP1]](%[[C0]], %[[C0]], %[[C0]])
// CHECK-DAG:   %[[LANEID:.+]] = affine.apply #[[MAP2]](%[[TX]], %[[TY]])
// CHECK-DAG:   %[[LANEOFF0:.+]] = affine.apply #[[MAP0]](%[[C0]], %[[LANEID]], %[[C0]])
// CHECK-DAG:   %[[LANEOFF1:.+]] = affine.apply #[[MAP1]](%[[C0]], %[[LANEID]], %[[C0]])
// CHECK-DAG:   %[[OFF0:.+]] = arith.addi %[[VECOFF0]], %[[C0]] : index
// CHECK-DAG:   %[[OFF1:.+]] = arith.addi %[[LANEOFF0]], %[[OFF0]] : index
// CHECK-DAG:   %[[OFF2:.+]] = arith.addi %[[VECOFF1]], %[[C0]] : index
// CHECK-DAG:   %[[OFF3:.+]] = arith.addi %[[LANEOFF1]], %[[OFF2]] : index
//     CHECK:   %[[LD0:.+]] = nvgpu.ldmatrix %{{.*}}[%[[OFF1]], %[[OFF3]]] {numTiles = 1 : i32, transpose = false} : memref<16x16xf16, #gpu.address_space<workgroup>> -> vector<1x2xf16>
//     CHECK:   %[[V0:.+]] = vector.insert_strided_slice %[[LD0]], %{{.*}} {offsets = [0, 0, 0, 0], strides = [1, 1]} : vector<1x2xf16> into vector<1x1x4x2xf16>
//     CHECK:   %[[VECOFF2:.+]] = affine.apply #[[MAP3]](%[[C0]], %[[C0]], %[[C0]])
//     CHECK:   %[[OFF4:.+]] = arith.addi %[[VECOFF2]], %[[C0]] : index
//     CHECK:   %[[OFF5:.+]] = arith.addi %[[LANEOFF1]], %[[OFF4]] : index
//     CHECK:   %[[LD1:.+]] = nvgpu.ldmatrix %{{.*}}[%[[OFF1]], %[[OFF5]]] {numTiles = 1 : i32, transpose = false} : memref<16x16xf16, #gpu.address_space<workgroup>> -> vector<1x2xf16>
//     CHECK:   %[[V1:.+]] = vector.insert_strided_slice %[[LD1]], %[[V0]] {offsets = [0, 0, 2, 0], strides = [1, 1]} : vector<1x2xf16> into vector<1x1x4x2xf16>
//     CHECK:   %[[VECOFF3:.+]] = affine.apply #[[MAP4]](%[[C0]], %[[C0]], %[[C0]])
//     CHECK:   %[[OFF6:.+]] = arith.addi %[[VECOFF3]], %[[C0]] : index
//     CHECK:   %[[OFF7:.+]] = arith.addi %[[LANEOFF0]], %[[OFF6]] : index
//     CHECK:   %[[LD2:.+]] = nvgpu.ldmatrix %{{.*}}[%[[OFF7]], %[[OFF3]]] {numTiles = 1 : i32, transpose = false} : memref<16x16xf16, #gpu.address_space<workgroup>> -> vector<1x2xf16>
//     CHECK:   %[[V2:.+]] = vector.insert_strided_slice %[[LD2]], %[[V1]] {offsets = [0, 0, 1, 0], strides = [1, 1]} : vector<1x2xf16> into vector<1x1x4x2xf16>
//     CHECK:   %[[LD3:.+]] = nvgpu.ldmatrix %{{.*}}[%[[OFF7]], %[[OFF5]]] {numTiles = 1 : i32, transpose = false} : memref<16x16xf16, #gpu.address_space<workgroup>> -> vector<1x2xf16>
//     CHECK:   %[[V3:.+]] = vector.insert_strided_slice %[[LD3]], %[[V2]] {offsets = [0, 0, 3, 0], strides = [1, 1]} : vector<1x2xf16> into vector<1x1x4x2xf16>
// CHECK-DAG:   %[[VECOFF2:.+]] = affine.apply #[[MAP5]](%[[C0]], %[[C0]], %[[C0]])
// CHECK-DAG:   %[[LANEOFF2:.+]] = affine.apply #[[MAP5]](%[[C0]], %[[LANEID]], %[[C0]])
// CHECK-DAG:   %[[OFF8:.+]] = arith.addi %[[VECOFF2]], %[[C0]] : index
// CHECK-DAG:   %[[OFF9:.+]] = arith.addi %[[LANEOFF1]], %[[OFF8]] : index
// CHECK-DAG:   %[[OFF10:.+]] = arith.addi %[[LANEOFF2]], %[[OFF2]] : index
//     CHECK:   %[[LD3:.+]] = nvgpu.ldmatrix %{{.*}}[%[[OFF10]], %[[OFF9]]] {numTiles = 1 : i32, transpose = true} : memref<8x16xf16, #gpu.address_space<workgroup>> -> vector<1x2xf16>
//     CHECK:   %[[V4:.+]] = vector.insert_strided_slice %[[LD3]], %{{.*}} {offsets = [0, 0, 0, 0], strides = [1, 1]} : vector<1x2xf16> into vector<1x1x2x2xf16>
//     CHECK:   %[[OFF11:.+]] = arith.addi %[[LANEOFF2]], %[[OFF4]] : index
//     CHECK:   %[[LD4:.+]] = nvgpu.ldmatrix %{{.*}}[%[[OFF11]], %[[OFF9]]] {numTiles = 1 : i32, transpose = true} : memref<8x16xf16, #gpu.address_space<workgroup>> -> vector<1x2xf16>
//     CHECK:   %[[V5:.+]] = vector.insert_strided_slice %[[LD4]], %[[V4]] {offsets = [0, 0, 1, 0], strides = [1, 1]} : vector<1x2xf16> into vector<1x1x2x2xf16>
//     CHECK:   %[[A:.+]] = vector.extract %[[V3]][0, 0] : vector<4x2xf16> from vector<1x1x4x2xf16>
//     CHECK:   %[[B:.+]] = vector.extract %[[V5]][0, 0] : vector<2x2xf16> from vector<1x1x2x2xf16>
//     CHECK:   nvgpu.mma.sync(%[[A]], %[[B]], %{{.*}}) {mmaShape = [16, 8, 16]} : (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf16>) -> vector<2x2xf16>

// -----

#map = affine_map<(d0) -> (d0 * 16)>
#map1 = affine_map<(d0, d1) -> (d1, d0)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d1)>
builtin.module attributes { transform.with_named_sequence } {
  func.func @matmul_dispatch_0_matmul_16x16x16_f16() {
    %c0 = arith.constant 0 : index
    %cst = arith.constant dense<0.000000e+00> : vector<16x16xf16>
    %c0_0 = arith.constant 0 : index
    %cst_1 = arith.constant 0.000000e+00 : f16
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0_0) flags(ReadOnly) : memref<16x16xf16>
    memref.assume_alignment %0, 64 : memref<16x16xf16>
    %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0_0) flags(ReadOnly) : memref<16x16xf16>
    memref.assume_alignment %1, 64 : memref<16x16xf16>
    %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0_0) flags(ReadOnly) : memref<16xf16>
    memref.assume_alignment %2, 64 : memref<16xf16>
    %3 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) alignment(64) offset(%c0_0) flags(ReadOnly) : memref<16xf16>
    memref.assume_alignment %3, 64 : memref<16xf16>
    %4 = hal.interface.binding.subspan set(0) binding(4) type(storage_buffer) alignment(64) offset(%c0_0) flags(ReadOnly) : memref<16x8xf16>
    memref.assume_alignment %4, 64 : memref<16x8xf16>
    %5 = hal.interface.binding.subspan set(0) binding(5) type(storage_buffer) alignment(64) offset(%c0_0) flags(ReadOnly) : memref<16x8xf16>
    memref.assume_alignment %5, 64 : memref<16x8xf16>
    %6 = hal.interface.binding.subspan set(0) binding(6) type(storage_buffer) alignment(64) offset(%c0_0) : memref<16x8xf16>
    memref.assume_alignment %6, 64 : memref<16x8xf16>
    %c1 = arith.constant 1 : index
    %c1_2 = arith.constant 1 : index
    %c1_3 = arith.constant 1 : index
    %workgroup_id_x = hal.interface.workgroup.id[0] : index
    %workgroup_count_x = hal.interface.workgroup.count[0] : index
    %workgroup_id_y = hal.interface.workgroup.id[1] : index
    %workgroup_count_y = hal.interface.workgroup.count[1] : index
    %workgroup_id_z = hal.interface.workgroup.id[2] : index
    %workgroup_count_z = hal.interface.workgroup.count[2] : index
    %c1_4 = arith.constant 1 : index
    %7 = affine.apply #map(%workgroup_id_x)
    %8 = vector.transfer_read %0[%7, %c0_0], %cst_1 {in_bounds = [true, true]} : memref<16x16xf16>, vector<16x16xf16>
    %9 = vector.transfer_read %1[%c0_0, %c0_0], %cst_1 {in_bounds = [true, true], permutation_map = #map1} : memref<16x16xf16>, vector<16x16xf16>
    %10 = vector.contract {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %8, %9, %cst : vector<16x16xf16>, vector<16x16xf16> into vector<16x16xf16>
    %11 = vector.transfer_read %2[%7], %cst_1 {in_bounds = [true]} : memref<16xf16>, vector<16xf16>
    %12 = vector.multi_reduction <maximumf>, %10, %11 [1] : vector<16x16xf16> to vector<16xf16>
    %13 = vector.transfer_read %3[%7], %cst_1 {in_bounds = [true]} : memref<16xf16>, vector<16xf16>
    %14 = arith.subf %11, %12 : vector<16xf16>
    %15 = math.exp %14 : vector<16xf16>
    %16 = arith.mulf %15, %13 : vector<16xf16>
    %17 = vector.broadcast %12 : vector<16xf16> to vector<16x16xf16>
    %18 = vector.transpose %17, [1, 0] : vector<16x16xf16> to vector<16x16xf16>
    %19 = arith.subf %10, %18 : vector<16x16xf16>
    %20 = math.exp %19 : vector<16x16xf16>
    %21 = vector.multi_reduction <add>, %20, %16 [1] : vector<16x16xf16> to vector<16xf16>
    %22 = vector.broadcast %21 : vector<16xf16> to vector<16x16xf16>
    %23 = vector.transpose %22, [1, 0] : vector<16x16xf16> to vector<16x16xf16>
    %24 = arith.divf %20, %23 : vector<16x16xf16>
    %subview = memref.subview %6[%7, 0] [16, 8] [1, 1] : memref<16x8xf16> to memref<16x8xf16, strided<[8, 1], offset: ?>>
    %25 = vector.broadcast %16 : vector<16xf16> to vector<8x16xf16>
    %26 = vector.broadcast %21 : vector<16xf16> to vector<8x16xf16>
    %27 = vector.transfer_read %5[%7, %c0_0], %cst_1 {in_bounds = [true, true]} : memref<16x8xf16>, vector<16x8xf16>
    %28 = arith.divf %25, %26 : vector<8x16xf16>
    %29 = vector.transpose %28, [1, 0] : vector<8x16xf16> to vector<16x8xf16>
    %30 = arith.mulf %29, %27 : vector<16x8xf16>
    %31 = vector.transfer_read %4[%c0_0, %c0_0], %cst_1 {in_bounds = [true, true], permutation_map = #map1} : memref<16x8xf16>, vector<8x16xf16>
    %32 = vector.contract {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %24, %31, %30 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
    vector.transfer_write %32, %subview[%c0_0, %c0_0] {in_bounds = [true, true]} : vector<16x8xf16>, memref<16x8xf16, strided<[8, 1], offset: ?>>
    return
  }
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    %reordered_func = transform.iree.reorder_transpose %top_level_func : (!transform.any_op) -> !transform.any_op
    transform.iree.apply_cse %reordered_func : !transform.any_op
    transform.yield
  }
} // module

// CHECK-DAG:  #[[MAP:.+]] = affine_map<(d0) -> (d0 * 16)>
// CHECK-DAG:  #[[MAP1:.+]] = affine_map<(d0, d1) -> (d1, d0)>
// CHECK-DAG:  #[[MAP2:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK-DAG:  #[[MAP3:.+]] = affine_map<(d0, d1, d2) -> (d1, d2)>
// CHECK-DAG:  #[[MAP4:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK:      func.func @matmul_dispatch_0_matmul_16x16x16_f16() {
// CHECK-DAG:    %[[CST:.+]] = arith.constant dense<0.000000e+00> : vector<16x16xf16>
// CHECK-DAG:    %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:    %[[CST_0:.+]] = arith.constant 0.000000e+00 : f16
// CHECK:        %[[D0:.+]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64)
// CHECK-SAME:     offset(%[[C0]]) flags(ReadOnly) : memref<16x16xf16>
// CHECK:        memref.assume_alignment %[[D0]], 64 : memref<16x16xf16>
// CHECK:        %[[D1:.+]] = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64)
// CHECK-SAME:     offset(%[[C0]]) flags(ReadOnly) : memref<16x16xf16>
// CHECK:        memref.assume_alignment %[[D1]], 64 : memref<16x16xf16>
// CHECK:        %[[D2:.+]] = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64)
// CHECK-SAME:     offset(%[[C0]]) flags(ReadOnly) : memref<16xf16>
// CHECK:        memref.assume_alignment %[[D2]], 64 : memref<16xf16>
// CHECK:        %[[D3:.+]] = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) alignment(64)
// CHECK-SAME:     offset(%[[C0]]) flags(ReadOnly) : memref<16xf16>
// CHECK:        memref.assume_alignment %[[D3]], 64 : memref<16xf16>
// CHECK:        %[[D4:.+]] = hal.interface.binding.subspan set(0) binding(4) type(storage_buffer) alignment(64)
// CHECK-SAME:     offset(%[[C0]]) flags(ReadOnly) : memref<16x8xf16>
// CHECK:        memref.assume_alignment %[[D4]], 64 : memref<16x8xf16>
// CHECK:        %[[D5:.+]] = hal.interface.binding.subspan set(0) binding(5) type(storage_buffer) alignment(64)
// CHECK-SAME:     offset(%[[C0]]) flags(ReadOnly) : memref<16x8xf16>
// CHECK:        memref.assume_alignment %[[D5]], 64 : memref<16x8xf16>
// CHECK:        %[[D6:.+]] = hal.interface.binding.subspan set(0) binding(6) type(storage_buffer) alignment(64)
// CHECK-SAME:     offset(%[[C0]]) : memref<16x8xf16>
// CHECK:        memref.assume_alignment %[[D6]], 64 : memref<16x8xf16>
// CHECK:        %[[WORKGROUP_ID_X:.+]] = hal.interface.workgroup.id[0] : index
// CHECK-DAG:    %[[D7:.+]] = affine.apply #[[MAP]](%[[WORKGROUP_ID_X]])
// CHECK:        %[[D8:.+]] = vector.transfer_read %[[D0]][%[[D7]], %[[C0]]], %[[CST_0]] {in_bounds = [true, true]} :
// CHECK-SAME:     memref<16x16xf16>, vector<16x16xf16>
// CHECK:        %[[D9:.+]] = vector.transfer_read %[[D1]][%[[C0]], %[[C0]]], %[[CST_0]] {in_bounds = [true, true],
// CHECK-SAME:     permutation_map = #[[MAP1]]} : memref<16x16xf16>, vector<16x16xf16>
// CHECK:        %[[D10:.+]] = vector.contract {indexing_maps = [#[[MAP2]], #[[MAP3]], #[[MAP4]]], iterator_types =
// CHECK-SAME:     ["parallel", "parallel", "reduction"], kind = #[[VECTOR:.+]].kind<add>} %[[D8]], %[[D9]], %[[CST]] :
// CHECK-SAME:     vector<16x16xf16>, vector<16x16xf16> into vector<16x16xf16>
// CHECK:        %[[D11:.+]] = vector.transfer_read %[[D2]][%[[D7]]], %[[CST_0]] {in_bounds = [true]} : memref<16xf16>,
// CHECK-SAME:     vector<16xf16>
// CHECK:        %[[D12:.+]] = vector.multi_reduction <maximumf>, %[[D10]], %[[D11]] [1] : vector<16x16xf16> to
// CHECK-SAME:     vector<16xf16>
// CHECK:        %[[D13:.+]] = vector.transfer_read %[[D3]][%[[D7]]], %[[CST_0]] {in_bounds = [true]} : memref<16xf16>,
// CHECK-SAME:     vector<16xf16>
// CHECK:        %[[D14:.+]] = arith.subf %[[D11]], %[[D12]] : vector<16xf16>
// CHECK:        %[[D15:.+]] = math.exp %[[D14]] : vector<16xf16>
// CHECK:        %[[D16:.+]] = arith.mulf %[[D15]], %[[D13]] : vector<16xf16>
// CHECK:        %[[D17:.+]] = vector.broadcast %[[D12]] : vector<16xf16> to vector<16x16xf16>
// CHECK:        %[[D18:.+]] = vector.transpose %[[D17]], [1, 0] : vector<16x16xf16> to vector<16x16xf16>
// CHECK:        %[[D19:.+]] = arith.subf %[[D10]], %[[D18]] : vector<16x16xf16>
// CHECK:        %[[D20:.+]] = math.exp %[[D19]] : vector<16x16xf16>
// CHECK:        %[[D21:.+]] = vector.multi_reduction <add>, %[[D20]], %[[D16]] [1] : vector<16x16xf16> to
// CHECK-SAME:     vector<16xf16>
// CHECK:        %[[D22:.+]] = vector.broadcast %[[D21]] : vector<16xf16> to vector<16x16xf16>
// CHECK:        %[[D23:.+]] = vector.transpose %[[D22]], [1, 0] : vector<16x16xf16> to vector<16x16xf16>
// CHECK:        %[[D24:.+]] = arith.divf %[[D20]], %[[D23]] : vector<16x16xf16>
// CHECK:        %[[SUBVIEW:.+]] = memref.subview %[[D6]][%[[D7]], 0] [16, 8] [1, 1] : memref<16x8xf16> to
// CHECK-SAME:     memref<16x8xf16, strided<[8, 1], offset: ?>>
// CHECK:        %[[D25:.+]] = vector.broadcast %[[D16]] : vector<16xf16> to vector<8x16xf16>
// CHECK:        %[[D26:.+]] = vector.broadcast %[[D21]] : vector<16xf16> to vector<8x16xf16>
// CHECK:        %[[D27:.+]] = vector.transfer_read %[[D5]][%[[D7]], %[[C0]]], %[[CST_0]] {in_bounds = [true, true]} :
// CHECK-SAME:     memref<16x8xf16>, vector<16x8xf16>
// CHECK:        %[[D28:.+]] = vector.transpose %[[D25]], [1, 0] : vector<8x16xf16> to vector<16x8xf16>
// CHECK:        %[[D29:.+]] = vector.transpose %[[D26]], [1, 0] : vector<8x16xf16> to vector<16x8xf16>
// CHECK:        %[[D30:.+]] = arith.divf %[[D28]], %[[D29]] : vector<16x8xf16>
// CHECK:        %[[D31:.+]] = arith.mulf %[[D30]], %[[D27]] : vector<16x8xf16>
// CHECK:        %[[D32:.+]] = vector.transfer_read %[[D4]][%[[C0]], %[[C0]]], %[[CST_0]] {in_bounds = [true, true],
// CHECK-SAME:     permutation_map = #[[MAP1]]} : memref<16x8xf16>, vector<8x16xf16>
// CHECK:        %[[D33:.+]] = vector.contract {indexing_maps = [#[[MAP2]], #[[MAP3]], #[[MAP4]]], iterator_types =
// CHECK-SAME:     ["parallel", "parallel", "reduction"], kind = #[[VECTOR]].kind<add>} %[[D24]], %[[D32]], %[[D31]] :
// CHECK-SAME:     vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
// CHECK:        vector.transfer_write %[[D33]], %[[SUBVIEW]][%[[C0]], %[[C0]]] {in_bounds = [true, true]} :
// CHECK-SAME:     vector<16x8xf16>, memref<16x8xf16, strided<[8, 1], offset: ?>>
// CHECK:        return
// CHECK:      }

// -----

#map = affine_map<(d0) -> (d0 * 16)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1)>
builtin.module attributes { transform.with_named_sequence } {
  func.func @double_matmul_dispatch_0_matmul_16x16x16() {
    %c0 = arith.constant 0 : index
    %cst = arith.constant dense<0.000000e+00> : vector<16x16xf16>
    %cst_0 = arith.constant dense<0.000000e+00> : vector<16x8xf16>
    %c0_1 = arith.constant 0 : index
    %cst_2 = arith.constant 0.000000e+00 : f16
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0_1) flags(ReadOnly) : memref<16x16xf16>
    memref.assume_alignment %0, 64 : memref<16x16xf16>
    %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0_1) flags(ReadOnly) : memref<16x16xf16>
    memref.assume_alignment %1, 64 : memref<16x16xf16>
    %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0_1) flags(ReadOnly) : memref<8x16xf16>
    memref.assume_alignment %2, 64 : memref<8x16xf16>
    %3 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) alignment(64) offset(%c0_1) : memref<16x8xf16>
    memref.assume_alignment %3, 64 : memref<16x8xf16>
    %c1 = arith.constant 1 : index
    %c1_3 = arith.constant 1 : index
    %c1_4 = arith.constant 1 : index
    %workgroup_id_x = hal.interface.workgroup.id[0] : index
    %workgroup_count_x = hal.interface.workgroup.count[0] : index
    %workgroup_id_y = hal.interface.workgroup.id[1] : index
    %workgroup_count_y = hal.interface.workgroup.count[1] : index
    %workgroup_id_z = hal.interface.workgroup.id[2] : index
    %workgroup_count_z = hal.interface.workgroup.count[2] : index
    %c1_5 = arith.constant 1 : index
    %4 = affine.apply #map(%workgroup_id_x)
    %5 = vector.transfer_read %0[%4, %c0_1], %cst_2 {in_bounds = [true, true]} : memref<16x16xf16>, vector<16x16xf16>
    %6 = vector.transfer_read %1[%c0_1, %c0_1], %cst_2 {in_bounds = [true, true]} : memref<16x16xf16>, vector<16x16xf16>
    %7 = vector.contract {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %5, %6, %cst : vector<16x16xf16>, vector<16x16xf16> into vector<16x16xf16>
    %subview = memref.subview %3[%4, 0] [16, 8] [1, 1] : memref<16x8xf16> to memref<16x8xf16, strided<[8, 1], offset: ?>>
    %8 = vector.transfer_read %2[%c0_1, %c0_1], %cst_2 {permutation_map = affine_map<(d0, d1) -> (d1, d0)>, in_bounds = [true, true]} : memref<8x16xf16>, vector<8x16xf16>
    %9 = vector.contract {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %7, %8, %cst_0 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
    vector.transfer_write %9, %subview[%c0_1, %c0_1] {in_bounds = [true, true]} : vector<16x8xf16>, memref<16x8xf16, strided<[8, 1], offset: ?>>
    return
  }
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    %transformed_func = transform.iree.layout_analysis_and_distribution %top_level_func : (!transform.any_op) -> (!transform.any_op)
    transform.yield
  }
} // module

// CHECK-DAG:  #[[MAP:.+]] = affine_map<(d0) -> (d0 * 16)>
// CHECK-DAG:  #[[MAP1:.+]] = affine_map<(d0, d1, d2) -> (d1 + d2 * 16)>
// CHECK-DAG:  #[[MAP2:.+]] = affine_map<(d0, d1, d2) -> (d0 * 2)>
// CHECK-DAG:  #[[MAP3:.+]] = affine_map<(d0, d1, d2) -> (d0 * 2 + 1)>
// CHECK-DAG:  #[[MAP4:.+]] = affine_map<(d0, d1, d2) -> (d0 * 2 + 8)>
// CHECK-DAG:  #[[MAP5:.+]] = affine_map<(d0, d1, d2) -> (d0 * 2 + 9)>
// CHECK-DAG:  #[[MAP6:.+]] = affine_map<(d0, d1, d2) -> (d1 + d2 * 16 + 8)>
// CHECK-DAG:  #[[MAP7:.+]] = affine_map<(d0, d1, d2) -> (d1 + d2 * 8)>
// CHECK-DAG:  #[[MAP8:.+]] = affine_map<(d0, d1, d2) -> (d1 + d2 * 8 + 8)>
// CHECK:      func.func @double_matmul_dispatch_0_matmul_16x16x16() {
// CHECK-DAG:    %[[CST:.+]] = arith.constant dense<0.000000e+00> : vector<1x2x2x2xf16>
// CHECK-DAG:    %[[CST_0:.+]] = arith.constant dense<0.000000e+00> : vector<1x1x2x2xf16>
// CHECK-DAG:    %[[C0:.+]] = arith.constant 0 : index
// CHECK:        %[[D0:.+]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64)
// CHECK-SAME:     offset(%[[C0]]) flags(ReadOnly) : memref<16x16xf16>
// CHECK:        memref.assume_alignment %[[D0]], 64 : memref<16x16xf16>
// CHECK:        %[[D1:.+]] = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64)
// CHECK-SAME:     offset(%[[C0]]) flags(ReadOnly) : memref<16x16xf16>
// CHECK:        memref.assume_alignment %[[D1]], 64 : memref<16x16xf16>
// CHECK:        %[[D2:.+]] = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64)
// CHECK-SAME:     offset(%[[C0]]) flags(ReadOnly) : memref<8x16xf16>
// CHECK:        memref.assume_alignment %[[D2]], 64 : memref<8x16xf16>
// CHECK:        %[[D3:.+]] = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) alignment(64)
// CHECK-SAME:     offset(%[[C0]]) : memref<16x8xf16>
// CHECK:        memref.assume_alignment %[[D3]], 64 : memref<16x8xf16>
// CHECK:        %[[WORKGROUP_ID_X:.+]] = hal.interface.workgroup.id[0] : index
// CHECK-DAG:    %[[D4:.+]] = affine.apply #[[MAP]](%[[WORKGROUP_ID_X]])
// CHECK-DAG:    %[[D5:.+]] = gpu.thread_id  x
// CHECK-DAG:    %[[D6:.+]] = gpu.thread_id  y
// CHECK-DAG:    %[[D7:.+]] = gpu.thread_id  z
// CHECK-DAG:    %[[CST_1:.+]] = arith.constant dense<0.000000e+00> : vector<1x1x4x2xf16>
// CHECK-DAG:    %[[D8:.+]] = affine.apply #[[MAP1]](%[[D5]], %[[D6]], %[[D7]])
// CHECK-DAG:    %[[D9:.+]] = affine.apply #[[MAP2]](%[[D5]], %[[D6]], %[[D7]])
// CHECK:        %[[D10:.+]] = arith.addi %[[D8]], %[[D4]] : index
// CHECK:        %[[D11:.+]] = arith.addi %[[D9]], %[[C0]] : index
// CHECK:        %[[D12:.+]] = memref.load %[[D0]][%[[D10]], %[[D11]]] : memref<16x16xf16>
// CHECK:        %[[D13:.+]] = vector.broadcast %[[D12]] : f16 to vector<1xf16>
// CHECK:        %[[D14:.+]] = vector.insert_strided_slice %[[D13]], %[[CST_1]] {offsets = [0, 0, 0, 0], strides = [1]}
// CHECK-SAME:     : vector<1xf16> into vector<1x1x4x2xf16>
// CHECK-DAG:    %[[D15:.+]] = affine.apply #[[MAP3]](%[[D5]], %[[D6]], %[[D7]])
// CHECK:        %[[D16:.+]] = arith.addi %[[D15]], %[[C0]] : index
// CHECK:        %[[D17:.+]] = memref.load %[[D0]][%[[D10]], %[[D16]]] : memref<16x16xf16>
// CHECK:        %[[D18:.+]] = vector.broadcast %[[D17]] : f16 to vector<1xf16>
// CHECK:        %[[D19:.+]] = vector.insert_strided_slice %[[D18]], %[[D14]] {offsets = [0, 0, 0, 1], strides = [1]} :
// CHECK-SAME:     vector<1xf16> into vector<1x1x4x2xf16>
// CHECK-DAG:    %[[D20:.+]] = affine.apply #[[MAP4]](%[[D5]], %[[D6]], %[[D7]])
// CHECK:        %[[D21:.+]] = arith.addi %[[D20]], %[[C0]] : index
// CHECK:        %[[D22:.+]] = memref.load %[[D0]][%[[D10]], %[[D21]]] : memref<16x16xf16>
// CHECK:        %[[D23:.+]] = vector.broadcast %[[D22]] : f16 to vector<1xf16>
// CHECK:        %[[D24:.+]] = vector.insert_strided_slice %[[D23]], %[[D19]] {offsets = [0, 0, 2, 0], strides = [1]} :
// CHECK-SAME:     vector<1xf16> into vector<1x1x4x2xf16>
// CHECK-DAG:    %[[D25:.+]] = affine.apply #[[MAP5]](%[[D5]], %[[D6]], %[[D7]])
// CHECK:        %[[D26:.+]] = arith.addi %[[D25]], %[[C0]] : index
// CHECK:        %[[D27:.+]] = memref.load %[[D0]][%[[D10]], %[[D26]]] : memref<16x16xf16>
// CHECK:        %[[D28:.+]] = vector.broadcast %[[D27]] : f16 to vector<1xf16>
// CHECK:        %[[D29:.+]] = vector.insert_strided_slice %[[D28]], %[[D24]] {offsets = [0, 0, 2, 1], strides = [1]} :
// CHECK-SAME:     vector<1xf16> into vector<1x1x4x2xf16>
// CHECK-DAG:    %[[D30:.+]] = affine.apply #[[MAP6]](%[[D5]], %[[D6]], %[[D7]])
// CHECK:        %[[D31:.+]] = arith.addi %[[D30]], %[[D4]] : index
// CHECK:        %[[D32:.+]] = memref.load %[[D0]][%[[D31]], %[[D11]]] : memref<16x16xf16>
// CHECK:        %[[D33:.+]] = vector.broadcast %[[D32]] : f16 to vector<1xf16>
// CHECK:        %[[D34:.+]] = vector.insert_strided_slice %[[D33]], %[[D29]] {offsets = [0, 0, 1, 0], strides = [1]} :
// CHECK-SAME:     vector<1xf16> into vector<1x1x4x2xf16>
// CHECK:        %[[D35:.+]] = memref.load %[[D0]][%[[D31]], %[[D16]]] : memref<16x16xf16>
// CHECK:        %[[D36:.+]] = vector.broadcast %[[D35]] : f16 to vector<1xf16>
// CHECK:        %[[D37:.+]] = vector.insert_strided_slice %[[D36]], %[[D34]] {offsets = [0, 0, 1, 1], strides = [1]} :
// CHECK-SAME:     vector<1xf16> into vector<1x1x4x2xf16>
// CHECK:        %[[D38:.+]] = memref.load %[[D0]][%[[D31]], %[[D21]]] : memref<16x16xf16>
// CHECK:        %[[D39:.+]] = vector.broadcast %[[D38]] : f16 to vector<1xf16>
// CHECK:        %[[D40:.+]] = vector.insert_strided_slice %[[D39]], %[[D37]] {offsets = [0, 0, 3, 0], strides = [1]} :
// CHECK-SAME:     vector<1xf16> into vector<1x1x4x2xf16>
// CHECK:        %[[D41:.+]] = memref.load %[[D0]][%[[D31]], %[[D26]]] : memref<16x16xf16>
// CHECK:        %[[D42:.+]] = vector.broadcast %[[D41]] : f16 to vector<1xf16>
// CHECK:        %[[D43:.+]] = vector.insert_strided_slice %[[D42]], %[[D40]] {offsets = [0, 0, 3, 1], strides = [1]} :
// CHECK-SAME:     vector<1xf16> into vector<1x1x4x2xf16>
// CHECK-DAG:    %[[CST_2:.+]] = arith.constant dense<0.000000e+00> : vector<2x1x2x2xf16>
// CHECK-DAG:    %[[D44:.+]] = affine.apply #[[MAP7]](%[[D5]], %[[D6]], %[[D7]])
// CHECK:        %[[D45:.+]] = arith.addi %[[D44]], %[[C0]] : index
// CHECK:        %[[D46:.+]] = memref.load %[[D1]][%[[D45]], %[[D11]]] : memref<16x16xf16>
// CHECK:        %[[D47:.+]] = vector.broadcast %[[D46]] : f16 to vector<1xf16>
// CHECK:        %[[D48:.+]] = vector.insert_strided_slice %[[D47]], %[[CST_2]] {offsets = [0, 0, 0, 0], strides = [1]}
// CHECK-SAME:     : vector<1xf16> into vector<2x1x2x2xf16>
// CHECK:        %[[D49:.+]] = memref.load %[[D1]][%[[D45]], %[[D16]]] : memref<16x16xf16>
// CHECK:        %[[D50:.+]] = vector.broadcast %[[D49]] : f16 to vector<1xf16>
// CHECK:        %[[D51:.+]] = vector.insert_strided_slice %[[D50]], %[[D48]] {offsets = [0, 0, 0, 1], strides = [1]} :
// CHECK-SAME:     vector<1xf16> into vector<2x1x2x2xf16>
// CHECK:        %[[D52:.+]] = memref.load %[[D1]][%[[D45]], %[[D21]]] : memref<16x16xf16>
// CHECK:        %[[D53:.+]] = vector.broadcast %[[D52]] : f16 to vector<1xf16>
// CHECK:        %[[D54:.+]] = vector.insert_strided_slice %[[D53]], %[[D51]] {offsets = [0, 0, 1, 0], strides = [1]} :
// CHECK-SAME:     vector<1xf16> into vector<2x1x2x2xf16>
// CHECK:        %[[D55:.+]] = memref.load %[[D1]][%[[D45]], %[[D26]]] : memref<16x16xf16>
// CHECK:        %[[D56:.+]] = vector.broadcast %[[D55]] : f16 to vector<1xf16>
// CHECK:        %[[D57:.+]] = vector.insert_strided_slice %[[D56]], %[[D54]] {offsets = [0, 0, 1, 1], strides = [1]} :
// CHECK-SAME:     vector<1xf16> into vector<2x1x2x2xf16>
// CHECK-DAG:    %[[D58:.+]] = affine.apply #[[MAP8]](%[[D5]], %[[D6]], %[[D7]])
// CHECK:        %[[D59:.+]] = arith.addi %[[D58]], %[[C0]] : index
// CHECK:        %[[D60:.+]] = memref.load %[[D1]][%[[D59]], %[[D11]]] : memref<16x16xf16>
// CHECK:        %[[D61:.+]] = vector.broadcast %[[D60]] : f16 to vector<1xf16>
// CHECK:        %[[D62:.+]] = vector.insert_strided_slice %[[D61]], %[[D57]] {offsets = [1, 0, 0, 0], strides = [1]} :
// CHECK-SAME:     vector<1xf16> into vector<2x1x2x2xf16>
// CHECK:        %[[D63:.+]] = memref.load %[[D1]][%[[D59]], %[[D16]]] : memref<16x16xf16>
// CHECK:        %[[D64:.+]] = vector.broadcast %[[D63]] : f16 to vector<1xf16>
// CHECK:        %[[D65:.+]] = vector.insert_strided_slice %[[D64]], %[[D62]] {offsets = [1, 0, 0, 1], strides = [1]} :
// CHECK-SAME:     vector<1xf16> into vector<2x1x2x2xf16>
// CHECK:        %[[D66:.+]] = memref.load %[[D1]][%[[D59]], %[[D21]]] : memref<16x16xf16>
// CHECK:        %[[D67:.+]] = vector.broadcast %[[D66]] : f16 to vector<1xf16>
// CHECK:        %[[D68:.+]] = vector.insert_strided_slice %[[D67]], %[[D65]] {offsets = [1, 0, 1, 0], strides = [1]} :
// CHECK-SAME:     vector<1xf16> into vector<2x1x2x2xf16>
// CHECK:        %[[D69:.+]] = memref.load %[[D1]][%[[D59]], %[[D26]]] : memref<16x16xf16>
// CHECK:        %[[D70:.+]] = vector.broadcast %[[D69]] : f16 to vector<1xf16>
// CHECK:        %[[D71:.+]] = vector.insert_strided_slice %[[D70]], %[[D68]] {offsets = [1, 0, 1, 1], strides = [1]} :
// CHECK-SAME:     vector<1xf16> into vector<2x1x2x2xf16>
// CHECK:        %[[D72:.+]] = vector.extract %[[CST]][0, 0] : vector<2x2xf16> from vector<1x2x2x2xf16>
// CHECK:        %[[D73:.+]] = vector.extract %[[D43]][0, 0] : vector<4x2xf16> from vector<1x1x4x2xf16>
// CHECK:        %[[D74:.+]] = vector.extract %[[D71]][0, 0] : vector<2x2xf16> from vector<2x1x2x2xf16>
// CHECK:        %[[D75:.+]] = nvgpu.mma.sync(%[[D73]], %[[D74]], %[[D72]]) {mmaShape = [16, 8, 16]} : (vector<4x2xf16>,
// CHECK-SAME:     vector<2x2xf16>, vector<2x2xf16>) -> vector<2x2xf16>
// CHECK:        %[[D76:.+]] = vector.insert %[[D75]], %[[CST]] [0, 0] : vector<2x2xf16> into vector<1x2x2x2xf16>
// CHECK:        %[[D77:.+]] = vector.extract %[[CST]][0, 1] : vector<2x2xf16> from vector<1x2x2x2xf16>
// CHECK:        %[[D78:.+]] = vector.extract %[[D71]][1, 0] : vector<2x2xf16> from vector<2x1x2x2xf16>
// CHECK:        %[[D79:.+]] = nvgpu.mma.sync(%[[D73]], %[[D78]], %[[D77]]) {mmaShape = [16, 8, 16]} : (vector<4x2xf16>,
// CHECK-SAME:     vector<2x2xf16>, vector<2x2xf16>) -> vector<2x2xf16>
// CHECK:        %[[D80:.+]] = vector.insert %[[D79]], %[[D76]] [0, 1] : vector<2x2xf16> into vector<1x2x2x2xf16>
// CHECK:        %[[SUBVIEW:.+]] = memref.subview %[[D3]][%[[D4]], 0] [16, 8] [1, 1] : memref<16x8xf16> to
// CHECK-SAME:     memref<16x8xf16, strided<[8, 1], offset: ?>>
// CHECK:        %[[D81:.+]] = memref.load %[[D2]][%[[D11]], %[[D45]]] : memref<8x16xf16>
// CHECK:        %[[D82:.+]] = vector.broadcast %[[D81]] : f16 to vector<1xf16>
// CHECK:        %[[D83:.+]] = vector.insert_strided_slice %[[D82]], %[[CST_0]] {offsets = [0, 0, 0, 0], strides = [1]}
// CHECK-SAME:     : vector<1xf16> into vector<1x1x2x2xf16>
// CHECK:        %[[D84:.+]] = memref.load %[[D2]][%[[D16]], %[[D45]]] : memref<8x16xf16>
// CHECK:        %[[D85:.+]] = vector.broadcast %[[D84]] : f16 to vector<1xf16>
// CHECK:        %[[D86:.+]] = vector.insert_strided_slice %[[D85]], %[[D83]] {offsets = [0, 0, 0, 1], strides = [1]} :
// CHECK-SAME:     vector<1xf16> into vector<1x1x2x2xf16>
// CHECK:        %[[D87:.+]] = memref.load %[[D2]][%[[D21]], %[[D45]]] : memref<8x16xf16>
// CHECK:        %[[D88:.+]] = vector.broadcast %[[D87]] : f16 to vector<1xf16>
// CHECK:        %[[D89:.+]] = vector.insert_strided_slice %[[D88]], %[[D86]] {offsets = [0, 0, 1, 0], strides = [1]} :
// CHECK-SAME:     vector<1xf16> into vector<1x1x2x2xf16>
// CHECK:        %[[D90:.+]] = memref.load %[[D2]][%[[D26]], %[[D45]]] : memref<8x16xf16>
// CHECK:        %[[D91:.+]] = vector.broadcast %[[D90]] : f16 to vector<1xf16>
// CHECK:        %[[D92:.+]] = vector.insert_strided_slice %[[D91]], %[[D89]] {offsets = [0, 0, 1, 1], strides = [1]} :
// CHECK-SAME:     vector<1xf16> into vector<1x1x2x2xf16>
// CHECK:        %[[D93:.+]] = vector.extract %[[D80]][0, 0] : vector<2x2xf16> from vector<1x2x2x2xf16>
// CHECK:        %[[D94:.+]] = vector.insert_strided_slice %[[D93]], %[[CST_1]] {offsets = [0, 0, 0, 0], strides = [1,
// CHECK-SAME:     1]} : vector<2x2xf16> into vector<1x1x4x2xf16>
// CHECK:        %[[D95:.+]] = vector.extract %[[D80]][0, 1] : vector<2x2xf16> from vector<1x2x2x2xf16>
// CHECK:        %[[D96:.+]] = vector.insert_strided_slice %[[D95]], %[[D94]] {offsets = [0, 0, 2, 0], strides = [1, 1]}
// CHECK-SAME:     : vector<2x2xf16> into vector<1x1x4x2xf16>
// CHECK:        %[[D97:.+]] = vector.extract %[[CST_0]][0, 0] : vector<2x2xf16> from vector<1x1x2x2xf16>
// CHECK:        %[[D98:.+]] = vector.extract %[[D96]][0, 0] : vector<4x2xf16> from vector<1x1x4x2xf16>
// CHECK:        %[[D99:.+]] = vector.extract %[[D92]][0, 0] : vector<2x2xf16> from vector<1x1x2x2xf16>
// CHECK:        %[[D100:.+]] = nvgpu.mma.sync(%[[D98]], %[[D99]], %[[D97]]) {mmaShape = [16, 8, 16]} :
// CHECK-SAME:     (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf16>) -> vector<2x2xf16>
// CHECK:        %[[D101:.+]] = vector.insert %[[D100]], %[[CST_0]] [0, 0] : vector<2x2xf16> into vector<1x1x2x2xf16>
// CHECK:        %[[D102:.+]] = vector.extract %[[D101]][0, 0, 0, 0] : f16 from vector<1x1x2x2xf16>
// CHECK:        %[[D103:.+]] = arith.addi %[[D8]], %[[C0]] : index
// CHECK:        memref.store %[[D102]], %[[SUBVIEW]][%[[D103]], %[[D11]]] : memref<16x8xf16, strided<[8, 1], offset:
// CHECK-SAME:     ?>>
// CHECK:        %[[D104:.+]] = vector.extract %[[D101]][0, 0, 0, 1] : f16 from vector<1x1x2x2xf16>
// CHECK:        memref.store %[[D104]], %[[SUBVIEW]][%[[D103]], %[[D16]]] : memref<16x8xf16, strided<[8, 1], offset:
// CHECK-SAME:     ?>>
// CHECK:        %[[D105:.+]] = vector.extract %[[D101]][0, 0, 1, 0] : f16 from vector<1x1x2x2xf16>
// CHECK:        %[[D106:.+]] = arith.addi %[[D30]], %[[C0]] : index
// CHECK:        memref.store %[[D105]], %[[SUBVIEW]][%[[D106]], %[[D11]]] : memref<16x8xf16, strided<[8, 1], offset:
// CHECK-SAME:     ?>>
// CHECK:        %[[D107:.+]] = vector.extract %[[D101]][0, 0, 1, 1] : f16 from vector<1x1x2x2xf16>
// CHECK:        memref.store %[[D107]], %[[SUBVIEW]][%[[D106]], %[[D16]]] : memref<16x8xf16, strided<[8, 1], offset:
// CHECK-SAME:     ?>>
// CHECK:        return
// CHECK:      }
