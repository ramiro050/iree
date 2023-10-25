// This example demonstrates calling dynamically imported functions in the
// runtime. Alternatively the functions can be embedded into the compiled IREE
// programs for hermetic deployment (see custom_dispatch/cpu/embedded/).

// NOTE: this file is identical to system_example.mlir besides the lit config
// controlling the iree-run-module flag.
// TODO(benvanik): find a way to share the files (environment variables saying
// what types to run, etc).

// RUN: iree-compile --iree-hal-target-backends=llvm-cpu %s | \
// RUN: iree-run-module \
// RUN:     --device=local-sync \
// RUN:     --executable_plugin=$IREE_BINARY_DIR/samples/custom_dispatch/xnnpack/plugin/system_plugin$IREE_DYLIB_EXT \
// RUN:     --module=- \
// RUN:     --function=mixed_invocation \
// RUN:     --input=2x4xf32=2 \
// RUN:     --input=2x4xf32=4 | \
// RUN: FileCheck %s --check-prefix=CHECK-SYSTEM

// CHECK-SYSTEM: EXEC @mixed_invocation
// simple_mul_workgroup
// CHECK-SYSTEM: size0_0=2, size0_1=4
// CHECK-SYSTEM: stride0_0=4, stride0_1=1
// CHECK-SYSTEM: size1_0=2, size1_1=4
// CHECK-SYSTEM: stride1_0=4, stride1_1=1
// CHECK-SYSTEM: size2_0=2, size2_1=4
// CHECK-SYSTEM: stride2_0=4, stride2_1=1
// CHECK-SYSTEM: dim0=2
// CHECK-SYSTEM: dim1=4
// CHECK-SYSTEM: processor_id=
// CHECK-SYSTEM: processor_data[0]=
// CHECK-SYSTEM: mul2[tid=0:index=0,0](2 * 4 = 8)
// CHECK-SYSTEM: mul2[tid=0:index=0,1](2 * 4 = 8)
// CHECK-SYSTEM: mul2[tid=0:index=0,2](2 * 4 = 8)
// CHECK-SYSTEM: mul2[tid=0:index=0,3](2 * 4 = 8)
// CHECK-SYSTEM: mul2[tid=0:index=1,0](2 * 4 = 8)
// CHECK-SYSTEM: mul2[tid=0:index=1,1](2 * 4 = 8)
// CHECK-SYSTEM: mul2[tid=0:index=1,2](2 * 4 = 8)
// CHECK-SYSTEM: mul2[tid=0:index=1,3](2 * 4 = 8)
// arith.addf 8 + 4 = 12
// CHECK-SYSTEM: 2x4xf32=[12 12 12 12][12 12 12 12]

module @example {

  // Executable containing exported shims and calls to external functions.
  // Each executable can contain multiple exported functions and variants for
  // different architectures or even devices. It's also possible to mix hand-
  // authored functions with code generated ones even for the same functions
  // such that code generation is used as a fallback when the hand-authored
  // kernels aren't supported at runtime.
  stream.executable private @executable {
    stream.executable.export public @simple_mul workgroups(%workload: index) -> (index, index, index) {
      // This host function is used to compute the XYZ workgroup count
      // dispatched at runtime. It can query the %device for capabilities
      // and limits (last-level cache sizes, etc). The other arguments are the
      // values passed in the dispatch operation (usually things like root
      // output op tensor dimensions and other abstract values).
      %x = affine.apply affine_map<()[s0] -> (s0 ceildiv 64)>()[%workload]
      %c1 = arith.constant 1 : index
      stream.return %x, %c1, %c1 : index, index, index
    }

    builtin.module {
      // External function declaration using a user-chosen calling convention.
      func.func private @simple_mul_workgroup(
          %binding0: memref<f32>,
          %binding0_offset : index,
          %binding0_size0 : index,
          %binding0_size1 : index,
          %binding0_stride0 : index,
          %binding0_stride1 : index,
          %binding1: memref<f32>,
          %binding1_offset : index,
          %binding1_size0 : index,
          %binding1_size1 : index,
          %binding1_stride0 : index,
          %binding1_stride1 : index,
          %binding2: memref<f32>,
          %binding2_offset : index,
          %binding2_size0 : index,
          %binding2_size1 : index,
          %binding2_stride0 : index,
          %binding2_stride1 : index,
          %dim0: index,
          %dim1: index,
          %tid : index) attributes {
        // We can include some additional fields on the parameters struct as
        // needed. Here we request which processor is executing the call and
        // its data fields as defined by runtime/src/iree/schemas/cpu_data.h.
        hal.import.fields = ["processor_id", "processor_data"],
        llvm.bareptr = true
      }

      // IREE exported function using stream bindings and operands.
      // Compiler passes will be able to optimize across this interface and
      // deduplicate bindings/operands, convert/pack operands, and inline
      // constants operands.
      func.func @simple_mul(
          %binding0: !stream.binding,
          %binding1: !stream.binding,
          %binding2: !stream.binding,
          %dim0: index,
          %dim1: index) {
        %c0 = arith.constant 0 : index

        // This function is invoked once per workgroup so determine where this
        // particular workgroup is in the grid. In this example we use a
        // workgroup size of 64x1x1 (which is exceedingly small for CPUs but
        // useful for demonstration).
        %workgroup_id_x = flow.dispatch.workgroup.id[0] : index
        %tid = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%workgroup_id_x]

        // Bindings are accessed by reference.
        %memref0 = stream.binding.subspan %binding0[%c0] : !stream.binding -> memref<?x?xf32>{%dim0, %dim1}
        %memref1 = stream.binding.subspan %binding1[%c0] : !stream.binding -> memref<?x?xf32>{%dim0, %dim1}
        %memref2 = stream.binding.subspan %binding2[%c0] : !stream.binding -> memref<?x?xf32>{%dim0, %dim1}

        %base0, %offset0, %sizes0:2, %strides0:2 = memref.extract_strided_metadata %memref0
            : memref<?x?xf32> -> memref<f32>, index, index, index, index, index
        %base1, %offset1, %sizes1:2, %strides1:2 = memref.extract_strided_metadata %memref1
            : memref<?x?xf32> -> memref<f32>, index, index, index, index, index
        %base2, %offset2, %sizes2:2, %strides2:2 = memref.extract_strided_metadata %memref2
            : memref<?x?xf32> -> memref<f32>, index, index, index, index, index
        

        // Call the externally defined C function with an (almost) plain C
        // calling convention. This will be fetched at runtime from the plugin binary.
        func.call @simple_mul_workgroup(
            %base0, %offset0, %sizes0#0, %sizes0#1, %strides0#0, %strides0#1,
            %base1, %offset1, %sizes1#0, %sizes1#1, %strides1#0, %strides1#1,
            %base2, %offset2, %sizes2#0, %sizes2#1, %strides2#0, %strides2#1,
            %dim0, %dim1,
            %workgroup_id_x)
            : (memref<f32>, index, index, index, index, index, memref<f32>, index, index, index, index, index, memref<f32>, index, index, index, index, index, index, index, index) -> ()

        // NOTE: this is code generated as normal - other MLIR ops can be used
        // here for looping/control flow, vector operations, linalg, etc.
        // This simple sample is just calling out to the external function but
        // microkernels fused with other code are possible.

        return
      }
    }
  }

  // Function demonstrating executable plugins and mixing plugins and codegen.
  // Invoke with:
  //  --device=local-sync
  //  --executable_plugin=system_plugin.so
  //  --function=mixed_invocation
  //  --input=2x4xf32=2
  //  --input=2x4xf32=4
  func.func @mixed_invocation(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {
    // The only externally available metadata in the dispatch are the values
    // passed in as operands. Here we pass in the dynamic dimension.
    %c0 = arith.constant 0 : index
    %dim0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
    %c1 = arith.constant 1 : index
    %dim1 = tensor.dim %arg0, %c1 : tensor<?x?xf32>

    // Dispatch a basic `ret = lhs * rhs` using an external function.
    // This form (@executable::@export) allows for automatic variant selection
    // when multi-targeting.
    %0 = flow.dispatch @executable::@simple_mul[%dim0](%arg0, %arg1, %dim0, %dim1) : (tensor<?x?xf32>{%dim0, %dim1}, tensor<?x?xf32>{%dim0, %dim1}, index, index) -> tensor<?x?xf32>{%dim0, %dim1}

    // Code gen some other ops - these will interleave with hand-authored
    // ones but naturally won't be able to fuse with them.
    %1 = arith.addf %0, %arg1 : tensor<?x?xf32>

    return %1 : tensor<?x?xf32>
  }

}  // module
