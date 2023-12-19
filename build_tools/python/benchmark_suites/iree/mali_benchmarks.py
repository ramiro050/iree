## Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Defines IREE Mali GPU benchmarks."""

from typing import List, Sequence

from benchmark_suites.iree import benchmark_presets, module_execution_configs, utils
from e2e_test_framework import unique_ids
from e2e_test_framework.definitions import common_definitions, iree_definitions
from e2e_test_framework.models import tflite_models, tf_models
from e2e_test_framework.device_specs import device_collections


class Android_Mali_Benchmarks(object):
    """Benchmarks on Android devices with Mali GPU."""

    ARM_VALHALL_GPU_TARGET = iree_definitions.CompileTarget(
        target_backend=iree_definitions.TargetBackend.VULKAN_SPIRV,
        target_architecture=common_definitions.DeviceArchitecture.ARM_VALHALL,
        target_abi=iree_definitions.TargetABI.VULKAN_ANDROID31,
    )
    DEFAULT_COMPILE_CONFIG = iree_definitions.CompileConfig.build(
        id=unique_ids.IREE_COMPILE_CONFIG_ANDROID_ARM_VALHALL_DEFAULTS,
        tags=["default-flags"],
        compile_targets=[ARM_VALHALL_GPU_TARGET],
    )
    EXPERIMENTAL_COMPILE_CONFIG = iree_definitions.CompileConfig.build(
        id=unique_ids.IREE_COMPILE_CONFIG_ANDROID_ARM_VALHALL_EXPERIMENTAL,
        tags=["experimental-flags", "fuse-padding", "max-concurrency"],
        compile_targets=[ARM_VALHALL_GPU_TARGET],
        extra_flags=[
            "--iree-flow-enable-fuse-padding-into-linalg-consumer-ops",
            "--iree-stream-partitioning-favor=max-concurrency",
        ],
    )
    # Kernel execution
    # Note that for kernel-execution benchmarks batch_size/repeat-count need to be
    # low enough that the whole dispatch completes within an OS-specific timeout.
    # Otherwise you'll get error like:
    # ```
    # INTERNAL; VK_ERROR_DEVICE_LOST; vkQueueSubmit; while invoking native function
    # hal.fence.await; while calling import;
    # ```
    EXPERIMENTAL_REPEATED_KERNEL_COMPILE_CONFIG = iree_definitions.CompileConfig.build(
        id=unique_ids.IREE_COMPILE_CONFIG_ANDROID_ARM_VALHALL_EXPERIMENTAL_REPEATED_KERNEL,
        tags=[
            "experimental-flags",
            "fuse-padding",
            "max-concurrency",
            "repeated-kernel",
        ],
        compile_targets=[ARM_VALHALL_GPU_TARGET],
        extra_flags=EXPERIMENTAL_COMPILE_CONFIG.extra_flags
        + ["--iree-hal-benchmark-dispatch-repeat-count=32"],
    )
    EXPERIMENTAL_REPEATED_KERNEL_RUN_FLAGS = ["--batch_size=32"]

    FP32_MODELS = [
        tflite_models.MOBILEBERT_FP32,
    ]
    FP16_MODELS = [tflite_models.MOBILEBERT_FP16]
    QUANT_MODELS = [
        tflite_models.MOBILEBERT_INT8,
    ]

    def generate(
        self,
    ) -> List[iree_definitions.E2EModelRunConfig]:
        default_gen_configs = self._get_module_generation_configs(
            compile_config=self.DEFAULT_COMPILE_CONFIG,
            fp32_models=self.FP32_MODELS,
            fp16_models=self.FP16_MODELS,
            quant_models=self.QUANT_MODELS,
        )
        experimental_gen_configs = self._get_module_generation_configs(
            compile_config=self.EXPERIMENTAL_COMPILE_CONFIG,
            fp32_models=self.FP32_MODELS,
            fp16_models=self.FP16_MODELS,
            quant_models=self.QUANT_MODELS,
        )
        experimental_repeated_kernel_gen_configs = self._get_module_generation_configs(
            compile_config=self.EXPERIMENTAL_REPEATED_KERNEL_COMPILE_CONFIG,
            fp32_models=self.FP32_MODELS,
            fp16_models=self.FP16_MODELS,
            quant_models=self.QUANT_MODELS,
        )

        mali_devices = device_collections.DEFAULT_DEVICE_COLLECTION.query_device_specs(
            architecture=common_definitions.DeviceArchitecture.ARM_VALHALL,
            host_environment=common_definitions.HostEnvironment.ANDROID_ARMV8_2_A,
        )
        run_configs = utils.generate_e2e_model_run_configs(
            module_generation_configs=default_gen_configs + experimental_gen_configs,
            module_execution_configs=[module_execution_configs.VULKAN_CONFIG],
            device_specs=mali_devices,
            presets=[benchmark_presets.ANDROID_GPU],
        )
        run_configs += utils.generate_e2e_model_run_configs(
            module_generation_configs=experimental_repeated_kernel_gen_configs,
            module_execution_configs=[
                module_execution_configs.VULKAN_BATCH_SIZE_32_CONFIG
            ],
            device_specs=mali_devices,
            presets=[benchmark_presets.ANDROID_GPU],
        )

        return run_configs

    def _get_module_generation_configs(
        self,
        compile_config: iree_definitions.CompileConfig,
        fp32_models: Sequence[common_definitions.Model],
        fp16_models: Sequence[common_definitions.Model],
        quant_models: Sequence[common_definitions.Model],
    ) -> List[iree_definitions.ModuleGenerationConfig]:
        demote_compile_config = iree_definitions.CompileConfig.build(
            id=compile_config.id + "-demote-f32-to-16",
            tags=compile_config.tags + ["demote-f32-to-f16"],
            compile_targets=compile_config.compile_targets,
            extra_flags=compile_config.extra_flags + ["--iree-opt-demote-f32-to-f16"],
        )
        return (
            [
                iree_definitions.ModuleGenerationConfig.build(
                    compile_config=compile_config,
                    imported_model=iree_definitions.ImportedModel.from_model(model),
                )
                for model in fp32_models
            ]
            + [
                iree_definitions.ModuleGenerationConfig.build(
                    compile_config=demote_compile_config,
                    imported_model=iree_definitions.ImportedModel.from_model(model),
                )
                for model in fp16_models
            ]
            + [
                iree_definitions.ModuleGenerationConfig.build(
                    compile_config=compile_config,
                    imported_model=iree_definitions.ImportedModel.from_model(model),
                )
                for model in quant_models
            ]
        )
