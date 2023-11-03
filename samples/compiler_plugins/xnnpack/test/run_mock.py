#!/usr/bin/env python3
# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# TODO: Turn this into a real test and wire it up.
# Usage:
#   iree-compile --iree-plugin=xnnpack_sample print.mlir -o print.vmfb
#   run_mock.py print.vmfb

import iree.runtime as rt
import sys

input_file = sys.argv[1]
print(f"--- Loading {input_file}")

with open(input_file, "rb") as f:
    vmfb_contents = f.read()


def create_xnnpack_module():
    class Xnnpack:
        def __init__(self, iface):
            ...

        def print_impl(self):
            print("+++ HELLO FROM XNNPACK")

    iface = rt.PyModuleInterface("xnnpack", Xnnpack)
    iface.export("print", "0v_v", Xnnpack.print_impl)
    return iface.create()


config = rt.Config("local-sync")
main_module = rt.VmModule.from_flatbuffer(config.vm_instance, vmfb_contents)
modules = config.default_vm_modules + (
    create_xnnpack_module(),
    main_module,
)
context = rt.SystemContext(vm_modules=modules, config=config)

print("--- Running main()")
context.modules.module.main()
