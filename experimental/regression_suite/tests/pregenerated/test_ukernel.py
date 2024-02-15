# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
from ireers import *

###############################################################################
# Fixtures
###############################################################################

COMMON_FLAGS = [
    "--iree-input-type=none",
]

argmax_ukernel_source = fetch_source_fixture(
    "https://storage.googleapis.com/shark_tank/ukernel_regression/20231217/argmax/argmax_3d_linalg.mlir",
    group="argmax_ukernel_linalg",
)


@pytest.fixture
def argmax_ukernel_host_cpu_vmfb(argmax_ukernel_source):
    return iree_compile(
        argmax_ukernel_source,
        "host_cpu",
        flags=COMMON_FLAGS
        + [
            "--iree-hal-target-backends=llvm-cpu",
            "--iree-llvmcpu-target-cpu-features=host",
        ],
    )


@pytest.fixture
def argmax_ukernel_gfx90a_rocm_vmfb(argmax_ukernel_source):
    return iree_compile(
        argmax_ukernel_source,
        "gfx90a_rocm",
        flags=COMMON_FLAGS
        + [
            "--iree-hal-target-backends=rocm",
            "--iree-rocm-target-chip=gfx90a",
            "--iree-rocm-link-bc=true",
            "--iree-rocm-enable-ukernels=argmax",
        ],
    )


@pytest.fixture
def argmax_ukernel_gfx940_rocm_vmfb(argmax_ukernel_source):
    return iree_compile(
        argmax_ukernel_source,
        "gfx940_rocm",
        flags=COMMON_FLAGS
        + [
            "--iree-hal-target-backends=rocm",
            "--iree-rocm-target-chip=gfx940",
            "--iree-rocm-link-bc=true",
            "--iree-rocm-enable-ukernels=argmax",
        ],
    )


###############################################################################
# Correctness
###############################################################################

# Generation script:
# argmax_input_f16 = np.random.normal(size=[2, 4, 33000]).astype(np.float32)
# argmax_output_f16 = np.argmax(argmax_input_f16,axis=-1).astype(np.float32)
# argmax_input_f32 = np.random.normal(size=[2, 4, 33000]).astype(np.float32)
# argmax_output_f32 = np.argmax(argmax_input_f32,axis=-1).astype(np.float32)
# TODO: Currently forcing sitofp (i32 -> f32) and (i64 -> f32) because expected_output
#       cannot compare signless i64 from vmfb and by default si64 from npy.

argmax_input_f16 = fetch_source_fixture(
    "https://storage.googleapis.com/shark_tank/ukernel_regression/20231217/argmax/argmax_3d_input_f16.npy",
    group="argmax_ukernel_input_f16",
)

argmax_output_f16 = fetch_source_fixture(
    "https://storage.googleapis.com/shark_tank/ukernel_regression/20231217/argmax/argmax_3d_output_f16.npy",
    group="argmax_ukernel_output_f16",
)

argmax_input_f32 = fetch_source_fixture(
    "https://storage.googleapis.com/shark_tank/ukernel_regression/20231217/argmax/argmax_3d_input_f32.npy",
    group="argmax_ukernel_input_f32",
)

argmax_output_f32 = fetch_source_fixture(
    "https://storage.googleapis.com/shark_tank/ukernel_regression/20231217/argmax/argmax_3d_output_f32.npy",
    group="argmax_ukernel_output_f32",
)


@pytest.mark.presubmit
@pytest.mark.unstable_linalg
@pytest.mark.plat_host_cpu
def test_correctness_host_cpu(
    argmax_ukernel_host_cpu_vmfb,
    argmax_input_f16,
    argmax_output_f16,
    argmax_input_f32,
    argmax_output_f32,
):
    iree_run_module(
        argmax_ukernel_host_cpu_vmfb,
        device="local-task",
        function="argmax_3d_dyn_f16i32",
        args=[
            f"--input=@{argmax_input_f16.path}",
            f"--expected_output=@{argmax_output_f16.path}",
        ],
    )
    iree_run_module(
        argmax_ukernel_host_cpu_vmfb,
        device="local-task",
        function="argmax_3d_dyn_f16i64",
        args=[
            f"--input=@{argmax_input_f16.path}",
            f"--expected_output=@{argmax_output_f16.path}",
        ],
    )

    iree_run_module(
        argmax_ukernel_host_cpu_vmfb,
        device="local-task",
        function="argmax_3d_dyn_f32i32",
        args=[
            f"--input=@{argmax_input_f32.path}",
            f"--expected_output=@{argmax_output_f32.path}",
        ],
    )
    iree_run_module(
        argmax_ukernel_host_cpu_vmfb,
        device="local-task",
        function="argmax_3d_dyn_f32i64",
        args=[
            f"--input=@{argmax_input_f32.path}",
            f"--expected_output=@{argmax_output_f32.path}",
        ],
    )


@pytest.mark.presubmit
@pytest.mark.unstable_linalg
@pytest.mark.plat_gfx90a_rocm
def test_correctness_gfx90a_rocm(
    argmax_ukernel_gfx90a_rocm_vmfb,
    argmax_input_f16,
    argmax_output_f16,
    argmax_input_f32,
    argmax_output_f32,
):
    iree_run_module(
        argmax_ukernel_gfx90a_rocm_vmfb,
        device="rocm",
        function="argmax_3d_dyn_f16i32",
        args=[
            f"--input=@{argmax_input_f16.path}",
            f"--expected_output=@{argmax_output_f16.path}",
        ],
    )
    iree_run_module(
        argmax_ukernel_gfx90a_rocm_vmfb,
        device="rocm",
        function="argmax_3d_dyn_f16i64",
        args=[
            f"--input=@{argmax_input_f16.path}",
            f"--expected_output=@{argmax_output_f16.path}",
        ],
    )

    iree_run_module(
        argmax_ukernel_gfx90a_rocm_vmfb,
        device="rocm",
        function="argmax_3d_dyn_f32i32",
        args=[
            f"--input=@{argmax_input_f32.path}",
            f"--expected_output=@{argmax_output_f32.path}",
        ],
    )
    iree_run_module(
        argmax_ukernel_gfx90a_rocm_vmfb,
        device="rocm",
        function="argmax_3d_dyn_f32i64",
        args=[
            f"--input=@{argmax_input_f32.path}",
            f"--expected_output=@{argmax_output_f32.path}",
        ],
    )


@pytest.mark.presubmit
@pytest.mark.unstable_linalg
@pytest.mark.plat_gfx940_rocm
def test_correctness_gfx940_rocm(
    argmax_ukernel_gfx940_rocm_vmfb,
    argmax_input_f16,
    argmax_output_f16,
    argmax_input_f32,
    argmax_output_f32,
):
    iree_run_module(
        argmax_ukernel_gfx940_rocm_vmfb,
        device="rocm",
        function="argmax_3d_dyn_f16i32",
        args=[
            f"--input=@{argmax_input_f16.path}",
            f"--expected_output=@{argmax_output_f16.path}",
        ],
    )
    iree_run_module(
        argmax_ukernel_gfx940_rocm_vmfb,
        device="rocm",
        function="argmax_3d_dyn_f16i64",
        args=[
            f"--input=@{argmax_input_f16.path}",
            f"--expected_output=@{argmax_output_f16.path}",
        ],
    )

    iree_run_module(
        argmax_ukernel_gfx940_rocm_vmfb,
        device="rocm",
        function="argmax_3d_dyn_f32i32",
        args=[
            f"--input=@{argmax_input_f32.path}",
            f"--expected_output=@{argmax_output_f32.path}",
        ],
    )
    iree_run_module(
        argmax_ukernel_gfx940_rocm_vmfb,
        device="rocm",
        function="argmax_3d_dyn_f32i64",
        args=[
            f"--input=@{argmax_input_f32.path}",
            f"--expected_output=@{argmax_output_f32.path}",
        ],
    )
