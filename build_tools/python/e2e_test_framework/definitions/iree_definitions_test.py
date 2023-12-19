## Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pathlib
import unittest

from e2e_test_framework.definitions import common_definitions, iree_definitions


class IreeDefinitionsTest(unittest.TestCase):
    def test_generate_run_flags(self):
        imported_model = iree_definitions.ImportedModel.from_model(
            common_definitions.Model(
                id="1234",
                name="tflite_m",
                tags=[],
                source_type=common_definitions.ModelSourceType.EXPORTED_TFLITE,
                source_url="https://example.com/xyz.tflite",
                entry_function="main",
                input_types=["1xf32", "2x2xf32"],
            )
        )
        execution_config = iree_definitions.ModuleExecutionConfig.build(
            id="123",
            tags=["test"],
            loader=iree_definitions.RuntimeLoader.EMBEDDED_ELF,
            driver=iree_definitions.RuntimeDriver.LOCAL_TASK,
            extra_flags=["--task=10"],
        )

        flags = iree_definitions.generate_run_flags(
            imported_model=imported_model,
            module_execution_config=execution_config,
        )

        self.assertEqual(
            flags,
            [
                "--function=main",
                "--task=10",
                "--device=local-task",
            ],
        )

    def test_generate_run_flags_with_cuda(self):
        imported_model = iree_definitions.ImportedModel.from_model(
            common_definitions.Model(
                id="1234",
                name="tflite_m",
                tags=[],
                source_type=common_definitions.ModelSourceType.EXPORTED_TFLITE,
                source_url="https://example.com/xyz.tflite",
                entry_function="main",
                input_types=["1xf32"],
            )
        )
        execution_config = iree_definitions.ModuleExecutionConfig.build(
            id="123",
            tags=["test"],
            loader=iree_definitions.RuntimeLoader.NONE,
            driver=iree_definitions.RuntimeDriver.CUDA,
            extra_flags=[],
        )

        flags = iree_definitions.generate_run_flags(
            imported_model=imported_model,
            module_execution_config=execution_config,
            gpu_id="3",
        )

        self.assertEqual(flags, ["--function=main", "--device=cuda://3"])

    def test_generate_run_flags_without_driver(self):
        imported_model = iree_definitions.ImportedModel.from_model(
            common_definitions.Model(
                id="1234",
                name="tflite_m",
                tags=[],
                source_type=common_definitions.ModelSourceType.EXPORTED_TFLITE,
                source_url="https://example.com/xyz.tflite",
                entry_function="main",
                input_types=["1xf32"],
            )
        )
        execution_config = iree_definitions.ModuleExecutionConfig.build(
            id="123",
            tags=["test"],
            loader=iree_definitions.RuntimeLoader.EMBEDDED_ELF,
            driver=iree_definitions.RuntimeDriver.LOCAL_TASK,
            extra_flags=["--task=10"],
        )

        flags = iree_definitions.generate_run_flags(
            imported_model=imported_model,
            module_execution_config=execution_config,
            with_driver=False,
        )

        self.assertEqual(flags, ["--function=main", "--task=10"])

    def test_materialize_run_flags(self):
        imported_model = iree_definitions.ImportedModel.from_model(
            common_definitions.Model(
                id="1234",
                name="tflite_m",
                tags=[],
                source_type=common_definitions.ModelSourceType.EXPORTED_TFLITE,
                source_url="https://example.com/xyz.tflite",
                entry_function="main",
                input_types=["1xf32", "2x2xf32"],
            )
        )
        compile_target = iree_definitions.CompileTarget(
            target_backend=iree_definitions.TargetBackend.CUDA,
            target_architecture=common_definitions.DeviceArchitecture.CUDA_SM80,
            target_abi=iree_definitions.TargetABI.LINUX_GNU,
        )
        compile_config = iree_definitions.CompileConfig(
            id="compile_config_a",
            name="compile_config_a",
            tags=["test"],
            compile_targets=[compile_target],
        )
        gen_config = iree_definitions.ModuleGenerationConfig.build(
            imported_model=imported_model, compile_config=compile_config
        )
        exec_config = iree_definitions.ModuleExecutionConfig.build(
            id="123",
            tags=["test"],
            loader=iree_definitions.RuntimeLoader.NONE,
            driver=iree_definitions.RuntimeDriver.CUDA,
        )
        device_spec = common_definitions.DeviceSpec.build(
            id="test_dev",
            device_name="test_model",
            host_environment=common_definitions.HostEnvironment.LINUX_X86_64,
            architecture=common_definitions.DeviceArchitecture.CUDA_SM80,
        )
        run_config = iree_definitions.E2EModelRunConfig.build(
            gen_config,
            exec_config,
            device_spec,
            input_data=common_definitions.DEFAULT_INPUT_DATA,
            tool=iree_definitions.E2EModelRunTool.IREE_BENCHMARK_MODULE,
        )

        inputs_dir = pathlib.PurePath("inputs_dir")
        flags = run_config.materialize_run_flags(gpu_id="10", inputs_dir=inputs_dir)

        self.assertIn("--device=cuda://10", flags)
        first_input = f'--input=@{inputs_dir / "input_0.npy"}'
        self.assertIn(first_input, flags)
        first_input_idx = flags.index(first_input)
        self.assertEqual(
            flags[first_input_idx + 1], f'--input=@{inputs_dir/"input_1.npy"}'
        )


class ModuleGenerationConfigTest(unittest.TestCase):
    def test_materialize_compile_flags(self):
        imported_model = iree_definitions.ImportedModel.from_model(
            common_definitions.Model(
                id="1234",
                name="tflite_m",
                tags=[],
                source_type=common_definitions.ModelSourceType.EXPORTED_TFLITE,
                source_url="https://example.com/xyz.tflite",
                entry_function="main",
                input_types=["1xf32"],
            )
        )
        compile_target = iree_definitions.CompileTarget(
            target_backend=iree_definitions.TargetBackend.LLVM_CPU,
            target_architecture=common_definitions.DeviceArchitecture.RV64_GENERIC,
            target_abi=iree_definitions.TargetABI.LINUX_GNU,
        )
        compile_config = iree_definitions.CompileConfig(
            id="compile_config_a",
            name="compile_config_a",
            tags=["test"],
            compile_targets=[compile_target],
            extra_flags=[r"--test=${MODULE_DIR}/test.json"],
        )
        gen_config = iree_definitions.ModuleGenerationConfig.build(
            imported_model=imported_model, compile_config=compile_config
        )

        flags = gen_config.materialize_compile_flags(
            module_dir_path=pathlib.Path("abc")
        )

        expected_path = pathlib.Path("abc", "test.json")
        self.assertIn(f"--test={expected_path}", flags)

    def test_materialize_compile_flags_invalid_module_dir_position(self):
        imported_model = iree_definitions.ImportedModel.from_model(
            common_definitions.Model(
                id="1234",
                name="tflite_m",
                tags=[],
                source_type=common_definitions.ModelSourceType.EXPORTED_TFLITE,
                source_url="https://example.com/xyz.tflite",
                entry_function="main",
                input_types=["1xf32"],
            )
        )
        compile_target = iree_definitions.CompileTarget(
            target_backend=iree_definitions.TargetBackend.LLVM_CPU,
            target_architecture=common_definitions.DeviceArchitecture.RV64_GENERIC,
            target_abi=iree_definitions.TargetABI.LINUX_GNU,
        )
        compile_config = iree_definitions.CompileConfig(
            id="compile_config_a",
            name="compile_config_a",
            tags=["test"],
            compile_targets=[compile_target],
            extra_flags=[r"--test=prefix/${MODULE_DIR}/test.json"],
        )
        gen_config = iree_definitions.ModuleGenerationConfig.build(
            imported_model=imported_model, compile_config=compile_config
        )
        expected_error = (
            r"^'\${MODULE_DIR}' needs to be the head of flag value if present,"
            r" but got 'prefix/\${MODULE_DIR}/test.json'.$"
        )

        self.assertRaisesRegex(
            ValueError,
            expected_error,
            lambda: gen_config.materialize_compile_flags(
                module_dir_path=pathlib.Path("abc")
            ),
        )


if __name__ == "__main__":
    unittest.main()
