// RUN: iree-opt --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(iree-llvmcpu-select-lowering-strategy, iree-llvmcpu-lower-executable-target)))' -split-input-file %s | FileCheck %s


#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable private @preset_config_generic_add  {
  hal.executable.variant @system_elf_x86_64 target(<"llvm-cpu", "system-elf-x86_64", {
    data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
    native_vector_size = 32 : index,
    target_triple = "x86_64-unknown-linux-gnu"
  }>) {
    hal.executable.export @mask_dynamic_generic_add layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2 : index, %arg3 : index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2, %arg3
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @mask_dynamic_generic_add() {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = hal.interface.constant.load[0] : i32
        %1 = hal.interface.constant.load[1] : i32
        %6 = arith.index_cast %0 : i32 to index
        %7 = arith.index_cast %1 : i32 to index
        %lhs_binding = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer)
            : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%6, %7}
        %rhs_binding = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer)
            : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%6, %7}
        %result_binding = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer)
            : !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>{%6, %7}
        %lhs = flow.dispatch.tensor.load %lhs_binding, offsets = [0, 0], sizes = [%6, %7], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%6, %7} -> tensor<?x?xf32>
        %rhs = flow.dispatch.tensor.load %rhs_binding, offsets = [0, 0], sizes = [%6, %7], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%6, %7} -> tensor<?x?xf32>
        %init = tensor.empty(%6, %7) : tensor<?x?xf32>
        %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<?x?xf32>) -> tensor<?x?xf32>
        %generic = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                                                    affine_map<(d0, d1) -> (d0, d1)>,
                                                    affine_map<(d0, d1) -> (d0, d1)>],
                                   iterator_types = ["parallel", "parallel"]}
          ins(%lhs, %rhs : tensor<?x?xf32>, tensor<?x?xf32>) outs(%fill : tensor<?x?xf32>) {
            ^bb0(%in0: f32, %in1: f32, %out: f32):
          %add = arith.addf %in0, %in1 : f32
          linalg.yield %add: f32
        } -> tensor<?x?xf32>
        flow.dispatch.tensor.store %generic, %result_binding, offsets = [0, 0], sizes = [%6, %7], strides = [1, 1]
            : tensor<?x?xf32> -> !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>{%6, %7}
        return
      }
    }
  }
}

// Masking is applied to the main vector loop when peeling is not used.

// CHECK-LABEL: func.func @mask_dynamic_generic_add
// Main loop
//         CHECK: scf.for
// CHECK-COUNT-2:   vector.maskedload
//         CHECK:   vector.maskedstore
// No epilogue
//     CHECK-NOT: scf.for

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
hal.executable private @preset_config_reduction  {
  hal.executable.variant @system_elf_x86_64 target(<"llvm-cpu", "system-elf-x86_64", {
    data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
    native_vector_size = 32 : index,
    target_triple = "x86_64-unknown-linux-gnu"
  }>) {
    hal.executable.export @mask_dynamic_reduction layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2 : index, %arg3 : index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2, %arg3
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @mask_dynamic_reduction() {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = hal.interface.constant.load[0] : i32
        %1 = hal.interface.constant.load[1] : i32
        %6 = arith.index_cast %0 : i32 to index
        %7 = arith.index_cast %1 : i32 to index
        %lhs_binding = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer)
            : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%6, %7}
        %result_binding = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer)
            : !flow.dispatch.tensor<writeonly:tensor<?xf32>>{%6}
        %lhs = flow.dispatch.tensor.load %lhs_binding, offsets = [0, 0], sizes = [%6, %7], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%6, %7} -> tensor<?x?xf32>
        %init = tensor.empty(%6) : tensor<?xf32>
        %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<?xf32>) -> tensor<?xf32>
        %generic = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                                                    affine_map<(d0, d1) -> (d0)>],
                                   iterator_types = ["parallel", "reduction"]}
          ins(%lhs : tensor<?x?xf32>) outs(%fill : tensor<?xf32>) {
            ^bb0(%in0: f32, %out: f32):
          %add = arith.addf %out, %in0 : f32
          linalg.yield %add: f32
        } -> tensor<?xf32>
        flow.dispatch.tensor.store %generic, %result_binding, offsets = [0], sizes = [%6], strides = [1]
            : tensor<?xf32> -> !flow.dispatch.tensor<writeonly:tensor<?xf32>>{%6}
        return
      }
    }
  }
}

//   CHECK-LABEL: func.func @mask_dynamic_reduction
//         CHECK:   vector.maskedload
//         CHECK:   vector.mask %{{.*}} { vector.reduction <add>

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable private @preset_config_generic_add  {
  hal.executable.variant @embedded_elf_rv32 target(<"llvm-cpu", "embedded-elf-riscv_32", {
      data_layout = "e-m:e-p:32:32-i64:64-n32-S128",
      native_vector_size = 32 : index,
      target_triple = "riscv32-none-elf"
    }>) {
    hal.executable.export @mask_dynamic_generic_add layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2 : index, %arg3 : index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2, %arg3
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @mask_dynamic_generic_add() {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = hal.interface.constant.load[0] : i32
        %1 = hal.interface.constant.load[1] : i32
        %6 = arith.index_cast %0 : i32 to index
        %7 = arith.index_cast %1 : i32 to index
        %lhs_binding = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer)
            : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%6, %7}
        %rhs_binding = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer)
            : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%6, %7}
        %result_binding = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer)
            : !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>{%6, %7}
        %lhs = flow.dispatch.tensor.load %lhs_binding, offsets = [0, 0], sizes = [%6, %7], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%6, %7} -> tensor<?x?xf32>
        %rhs = flow.dispatch.tensor.load %rhs_binding, offsets = [0, 0], sizes = [%6, %7], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%6, %7} -> tensor<?x?xf32>
        %init = tensor.empty(%6, %7) : tensor<?x?xf32>
        %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<?x?xf32>) -> tensor<?x?xf32>
        %generic = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                                                    affine_map<(d0, d1) -> (d0, d1)>,
                                                    affine_map<(d0, d1) -> (d0, d1)>],
                                   iterator_types = ["parallel", "parallel"]}
          ins(%lhs, %rhs : tensor<?x?xf32>, tensor<?x?xf32>) outs(%fill : tensor<?x?xf32>) {
            ^bb0(%in0: f32, %in1: f32, %out: f32):
          %add = arith.addf %in0, %in1 : f32
          linalg.yield %add: f32
        } -> tensor<?x?xf32>
        flow.dispatch.tensor.store %generic, %result_binding, offsets = [0, 0], sizes = [%6, %7], strides = [1, 1]
            : tensor<?x?xf32> -> !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>{%6, %7}
        return
      }
    }
  }
}

// Masking is applied to the main vector loop when peeling is not used.

// CHECK-LABEL: func.func @mask_dynamic_generic_add
// Main loop
//         CHECK: scf.for
// CHECK-COUNT-2:   vector.maskedload
//         CHECK:   vector.maskedstore
// No epilogue
//     CHECK-NOT: scf.for

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable private @preset_config_generic_add  {
  hal.executable.variant @embedded_elf_rv32 target(<"llvm-cpu", "embedded-elf-arm_64", {
    data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
    native_vector_size = 16 : index,
    target_triple = "aarch64-none-elf"
  }>) {
    hal.executable.export @mask_dynamic_generic_add layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2 : index, %arg3 : index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2, %arg3
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @mask_dynamic_generic_add() {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = hal.interface.constant.load[0] : i32
        %1 = hal.interface.constant.load[1] : i32
        %6 = arith.index_cast %0 : i32 to index
        %7 = arith.index_cast %1 : i32 to index
        %lhs_binding = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer)
            : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%6, %7}
        %rhs_binding = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer)
            : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%6, %7}
        %result_binding = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer)
            : !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>{%6, %7}
        %lhs = flow.dispatch.tensor.load %lhs_binding, offsets = [0, 0], sizes = [%6, %7], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%6, %7} -> tensor<?x?xf32>
        %rhs = flow.dispatch.tensor.load %rhs_binding, offsets = [0, 0], sizes = [%6, %7], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%6, %7} -> tensor<?x?xf32>
        %init = tensor.empty(%6, %7) : tensor<?x?xf32>
        %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<?x?xf32>) -> tensor<?x?xf32>
        %generic = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                                                    affine_map<(d0, d1) -> (d0, d1)>,
                                                    affine_map<(d0, d1) -> (d0, d1)>],
                                   iterator_types = ["parallel", "parallel"]}
          ins(%lhs, %rhs : tensor<?x?xf32>, tensor<?x?xf32>) outs(%fill : tensor<?x?xf32>) {
            ^bb0(%in0: f32, %in1: f32, %out: f32):
          %add = arith.addf %in0, %in1 : f32
          linalg.yield %add: f32
        } -> tensor<?x?xf32>
        flow.dispatch.tensor.store %generic, %result_binding, offsets = [0, 0], sizes = [%6, %7], strides = [1, 1]
            : tensor<?x?xf32> -> !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>{%6, %7}
        return
      }
    }
  }
}

// Masking should not happen on aarch64 if there is no SVE support.

// CHECK-LABEL: func.func @mask_dynamic_generic_add
//   CHECK-NOT:   vector.maskedload

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>,
    #hal.descriptor_set.binding<3, storage_buffer>
  ]>
]>
hal.executable private @preset_config_matmul  {
  hal.executable.variant @llvm target(<"llvm-cpu", "embedded-elf-arm_64", {
    data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
    cpu_features = "+sve",
    native_vector_size = 16 : index,
    target_triple = "aarch64-none-elf"
  }>) {
    hal.executable.export @mask_matmul_sve layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2 : index, %arg3 : index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2, %arg3
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @mask_matmul_sve() {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %M = hal.interface.constant.load[0] : index
        %N = hal.interface.constant.load[1] : index
        %K = hal.interface.constant.load[2] : index
        %lhs_binding = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer)
            : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%M, %K}
        %rhs_binding = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer)
            : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%K, %N}
        %init_binding = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer)
            : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%M, %N}
        %result_binding = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer)
            : !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>{%M, %N}
              %lhs = flow.dispatch.tensor.load %lhs_binding, offsets = [0, 0], sizes = [%M, %K], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%M, %K} -> tensor<?x?xf32>
        %rhs = flow.dispatch.tensor.load %rhs_binding, offsets = [0, 0], sizes = [%K, %N], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%K, %N} -> tensor<?x?xf32>
        %init = flow.dispatch.tensor.load %init_binding, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%M, %N} -> tensor<?x?xf32>
        %gemm = linalg.matmul ins(%lhs, %rhs : tensor<?x?xf32>, tensor<?x?xf32>) outs(%init : tensor<?x?xf32>) -> tensor<?x?xf32>
        flow.dispatch.tensor.store %gemm, %result_binding, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
            : tensor<?x?xf32> -> !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>{%M, %N}
        return
      }
    }
  }
}

// Masking is applied to the matmul on aarch64 when SVE is enabled.

// CHECK-LABEL: func.func @mask_matmul_sve
// CHECK:   vector.maskedload

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable private @preset_config_generic_add  {
  hal.executable.variant @embedded_elf_arm_64 target(<"llvm-cpu", "embedded-elf-arm_64", {
    cpu_features = "+sve",
    data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
    native_vector_size = 16 : index,
    target_triple = "aarch64-none-elf"
  }>) {
    hal.executable.export @mask_dynamic_generic_add layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2 : index, %arg3 : index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2, %arg3
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @mask_dynamic_generic_add() {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = hal.interface.constant.load[0] : i32
        %1 = hal.interface.constant.load[1] : i32
        %6 = arith.index_cast %0 : i32 to index
        %7 = arith.index_cast %1 : i32 to index
        %lhs_binding = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer)
            : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%6, %7}
        %rhs_binding = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer)
            : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%6, %7}
        %result_binding = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer)
            : !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>{%6, %7}
        %lhs = flow.dispatch.tensor.load %lhs_binding, offsets = [0, 0], sizes = [%6, %7], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%6, %7} -> tensor<?x?xf32>
        %rhs = flow.dispatch.tensor.load %rhs_binding, offsets = [0, 0], sizes = [%6, %7], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%6, %7} -> tensor<?x?xf32>
        %init = tensor.empty(%6, %7) : tensor<?x?xf32>
        %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<?x?xf32>) -> tensor<?x?xf32>
        %generic = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                                                    affine_map<(d0, d1) -> (d0, d1)>,
                                                    affine_map<(d0, d1) -> (d0, d1)>],
                                   iterator_types = ["parallel", "parallel"]}
          ins(%lhs, %rhs : tensor<?x?xf32>, tensor<?x?xf32>) outs(%fill : tensor<?x?xf32>) {
            ^bb0(%in0: f32, %in1: f32, %out: f32):
          %add = arith.addf %in0, %in1 : f32
          linalg.yield %add: f32
        } -> tensor<?x?xf32>
        flow.dispatch.tensor.store %generic, %result_binding, offsets = [0, 0], sizes = [%6, %7], strides = [1, 1]
            : tensor<?x?xf32> -> !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>{%6, %7}
        return
      }
    }
  }
}

// Masking is applied to the peeled loop on aarch64 when SVE is enabled.

// CHECK-LABEL: func.func @mask_dynamic_generic_add
// Main loop
//         CHECK: scf.for
// Peeled loop:
//         CHECK: scf.for
// CHECK-COUNT-2:   vector.maskedload
//         CHECK:   vector.maskedstore
