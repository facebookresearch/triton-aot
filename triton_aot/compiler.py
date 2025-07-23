# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#!/usr/bin/env python3


import copy
import glob
import hashlib
import importlib
import json
import logging
import multiprocessing as mp
import os
import shlex
import signal
import subprocess
import textwrap
from collections import Counter, namedtuple
from typing import Any

import torch
import triton
import triton.compiler

from triton.compiler.compiler import ASTSource
from triton.runtime.jit import JITFunction

try:
    # This no longer exists in the Triton Update.
    # pyre-ignore[21]: Could not find a name `AttrsDescriptor` defined in module `triton.compiler.compiler`.
    from triton.compiler.compiler import AttrsDescriptor
except ImportError:
    AttrsDescriptor = None

logger: logging.Logger = logging.getLogger(__name__)


IS_AMD = torch.version.hip is not None

"""
TODO (ginzburg): Figure out if we can query the AMD backend in Triton without having the driver loaded
def get_backend_options():
    driver = triton.runtime.driver
    target = driver.active.get_current_target()
    backend = triton.compiler.compiler.make_backend(target)
    options = backend.parse_options(dict())
    return options.__dict__


def get_backend_num_stages():
    options = get_backend_options()
    return options.get("num_stages", 2 if torch.version.hip else 3)
"""

TRITON_VERSION = triton.__version__
AUTOTUNE_ATTRs = {
    "num_warps": 4,
    "num_stages": 3,
    # AMD only
    "matrix_instr_nonkdim": 0,
    "waves_per_eu": 0,
    "kpack": 1,
}


class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, set):
            return {"__set__": True, "items": sorted(obj)}
        # Handle other non-serializable types
        return super().default(obj)


def hash_spec(spec):
    serialized_dict = json.dumps(spec, cls=CustomEncoder, sort_keys=True)
    return hashlib.sha256(serialized_dict.encode("utf-8")).hexdigest()


instance_descriptor = namedtuple(
    "instance_descriptor",
    [
        "divisible_by_16",
        "equal_to_1",
        "ids_of_folded_args",
        "divisible_by_8",
    ],
)

CTYPES = {
    "i1": "bool",
    "u8": "uint8_t",
    "i8": "int8_t",
    "i16": "int16_t",
    "i32": "int32_t",
    "i64": "int64_t",
    "fp16": "half",
    "fp32": "float",
    "fp64": "double",
    "bf16": "__nv_bfloat16",
}

ATYPES = {
    "*i1": "at::kBool",
    "*u8": "at::kByte",
    "*i8": "at::kChar",
    "*i16": "at::kShort",
    "*i32": "at::kInt",
    "*i64": "at::kLong",
    "*fp16": "at::kHalf",
    "*fp32": "at::kFloat",
    "*fp64": "at::kDouble",
    "*bf16": "at::kBFloat16",
}

PY_TYPES_TO_CPP_TYPES = {
    int: "int64_t",
    str: "at::string",
    float: "double",
}


HEURISTIC_TYPES = (
    triton.runtime.autotuner.Heuristics,
    triton.runtime.autotuner.Autotuner,
)

HIP_CC_TO_ARCH_INFO = {
    90: "gfx90a",
    94: "gfx942",
}


def get_warp_size():
    try:
        from triton.runtime import driver

        # for now, it only works for AMD Triton after the pin update
        # https://fburl.com/code/2v682rhh
        _, _, warp_size = driver.active.get_current_target()
        return warp_size
    except Exception:
        if IS_AMD:
            # `gfx9` archs, warp_size is 64, ref: https://fburl.com/code/zqnzd95z
            return 64
        else:
            # use 32 for all NV archs, ref: https://fburl.com/code/yzj0gwai
            return 32


def unwrap_heuristic(func):
    while not isinstance(func, triton.runtime.jit.JITFunction):
        func = func.fn
    return func


def gen_kernel_name(
    fn,
    *,
    cc=None,
    signature=None,
    constants=None,
    configs=None,
    num_warps=4,
    num_stages=3,
    **kwargs,
):
    name = fn.__name__
    sig = "_".join([p.replace("*", "p") for p in signature.values()])
    const = "_".join(map(str, constants.values()))
    cc = f"sm{cc}"
    autotune_configs = []
    autotune_configs.append(f"w{num_warps}")
    autotune_configs.append(f"s{num_stages}")
    # AMD only
    if "matrix_instr_nonkdim" in kwargs:
        autotune_configs.append(f"matrix{kwargs['matrix_instr_nonkdim']}")
    if "waves_per_eu" in kwargs:
        autotune_configs.append(f"wave{kwargs['waves_per_eu']}")
    if "kpack" in kwargs:
        autotune_configs.append(f"kpack{kwargs['kpack']}")
    # See kernel_suffix in triton/compiler/code_generator.py
    suffix = ""
    for i, _ in enumerate(signature):
        suffix += str(i)
        if i in configs[0].equal_to_1:
            suffix += "c"
        if i in configs[0].divisible_by_16:
            suffix += "d"
        if i in configs[0].divisible_by_8:
            suffix += "e"
    return "_".join([name, cc, sig, const] + autotune_configs + [suffix])


def hash_kernel_name(kernel_name):
    return "kernel_" + hashlib.sha1(kernel_name.encode("utf-8")).hexdigest()


def gen_cubin(kernel_name, kernel, install_dir, *, objcopy=None):
    hashed = hash_kernel_name(kernel_name)
    if IS_AMD:
        binary_file = f"{install_dir}/{hashed}.hsaco"
        with open(binary_file, "wb") as hsaco:
            hsaco.write(kernel.asm["hsaco"])
        o_file = f"{install_dir}/{hashed}.hsaco.o"
        source_symbol_name = f"_binary_{hashed}_hsaco_start"
        # we re-use the _cubin suffix for AMD, otherwise, we need to diverge the whole codebase
        target_symbol_name = f"{kernel_name}_cubin"
    else:
        binary_file = f"{install_dir}/{hashed}.cubin"
        with open(binary_file, "wb") as cubin:
            cubin.write(kernel.asm["cubin"])
        o_file = f"{install_dir}/{hashed}.cubin.o"
        source_symbol_name = f"_binary_{hashed}_cubin_start"
        target_symbol_name = f"{kernel_name}_cubin"

    if objcopy is not None:
        subprocess.run(
            shlex.split(objcopy)
            + [
                "--input",
                "binary",
                os.path.basename(binary_file),
                os.path.basename(o_file),
                "--redefine-sym",
                f"{source_symbol_name}={target_symbol_name}",
                "--rename-section",
                ".data=.triton",
            ],
            cwd=install_dir,
            check=True,
        )
        return f'extern "C" {{ extern unsigned char {target_symbol_name}[]; }}'
    else:
        with open(binary_file, "rb") as binary:
            binary_bytes = binary.read()
            # Convert cubin to hex format suitable for include.
            hexdump = [f"0x{x}" for x in binary_bytes.hex(",").split(",")]
            hexdump = textwrap.indent(textwrap.fill(", ".join(hexdump), 80), "  ")
            return f"unsigned char {target_symbol_name}[] = {{\n{hexdump}\n}};"


def gen_loader(kernel_name, cubin_name, shared):
    return textwrap.dedent(
        f"""
        CUfunction load_{kernel_name}(void)
        {{
            thread_local std::unordered_map<c10::DeviceIndex, CUfunction> cache;
            auto idx = c10::cuda::current_device();
            auto res = cache.find(idx);
            if (res != cache.end()) {{
                return res->second;
            }}
            CUfunction func;
            CUmodule mod_ptr;
            CUresult err;
            void *image = (void *)&{kernel_name}_cubin;
            err = cuModuleLoadData(&mod_ptr, image);
            if (err != 0) {{
                printf("cuModuleLoadData returned error: %d for {kernel_name}\\n", err);
                return NULL;
            }}
            err = cuModuleGetFunction(&func, mod_ptr, "{cubin_name}");
            if (err != 0) {{
                printf("cuModuleGetFunction returned error: %d for {kernel_name}\\n", err);
                return NULL;
            }}

            check_errors({shared}, func);
            cache.emplace(idx, func);
            return func;
        }}
    """
    )


def gen_launcher_params(kernel_name, func, *, signature=None, constants=None, **kwargs):
    args = ["gridDims grid"]
    for i, arg in enumerate(func.arg_names):
        if i in signature:
            ttype = signature[i]
            if ttype.startswith("*"):
                ctype = "void*"
            else:
                ctype = CTYPES[ttype]
            args.append(f"{ctype} {arg}")
            continue
    arg_string = ", ".join(args)
    return arg_string


def gen_launch_args(func, *, signature=None, constants=None, configs=None, **kwargs):
    args = []
    for i, arg in enumerate(func.arg_names):
        if i in constants:
            continue
        assert i in signature, f"Argument {i} ({arg}) does not appear in signature"
        args.append(f"&{arg}")
    return ", ".join(args)


def gen_launcher(kernel_name, func, kernel, shared, **kwargs):
    params = gen_launcher_params(kernel_name, func, **kwargs)
    args = gen_launch_args(func, **kwargs)

    return textwrap.dedent(
        f"""
        void {kernel_name}({params}) {{
            CUfunction func = load_{kernel_name}();
            CUstream stream = grid.stream ? grid.stream : c10::cuda::getCurrentCUDAStream().stream();
            void *args[] = {{ {args} }};
            auto res = cuLaunchKernel(func, grid.x, grid.y, grid.z, {get_warp_size()} * {kwargs['num_warps']}, 1, 1, {shared}, stream, args, NULL);
            AT_CUDA_DRIVER_CHECK(res);
        }}
    """
    )


def gen_selector_params_inner(
    func, *, signature=None, constants=None, optional=None, **kwargs
):
    args = ["gridDims grid"]
    for i, arg in enumerate(func.arg_names):
        if i in signature:
            if signature[i].startswith("*"):
                if (
                    i in optional
                ):  # handle the case where we have an optional input and has value
                    args.append(f"const std::optional<at::Tensor>& {arg}")
                else:
                    args.append(f"at::Tensor& {arg}")
                continue
            ctype = CTYPES[signature[i]]
            args.append(f"{ctype} {arg}")
        elif i in constants:
            if isinstance(constants[i], bool):
                args.append(f"bool {arg}")
            elif isinstance(constants[i], int):
                args.append(f"int {arg}")
            elif isinstance(constants[i], str):
                args.append(f"const std::string& {arg}")
            elif (
                constants[i] is None
            ):  # handle the case where we have an optional value input but doesn't have a value
                args.append(f"const std::optional<at::Tensor>& {arg}")
            else:
                raise AssertionError(
                    f"Unsupported type {type(constants[i])} for {arg}."
                )
        else:
            raise AssertionError(f"Unspecified argument {arg}")

    for name, value in AUTOTUNE_ATTRs.items():
        args.append(f"{type(value).__name__} {name}")
    return ", ".join(args)


def gen_selector_params(func, specs):
    # TODO: Check consistency across all specs
    return gen_selector_params_inner(func, **specs[0])


def gen_launcher_call_args(func, *, signature=None, optional=None, **kwargs):
    args = ["grid"]
    for i, arg in enumerate(func.arg_names):
        if i in signature:
            if signature[i].startswith("*"):
                if i in optional:
                    args.append(f"{arg}.value().data_ptr()")
                else:
                    args.append(f"{arg}.data_ptr()")
            else:
                args.append(arg)
    return ", ".join(args)


def gen_guarded_calls(func, specs):
    calls = []
    for spec in specs:
        kernel_name = gen_kernel_name(func, **spec)
        args = gen_launcher_call_args(func, **spec)
        guards = ""

        # Guard on compute capability.
        if "cc" in spec:
            guards += f"if (cc == {spec['cc']}) "

        # Guard on tensor dtypes
        for i, ttype in spec["signature"].items():
            if ttype.startswith("*"):
                arg = func.arg_names[i]
                atype = ATYPES[ttype]
                if i in spec["optional"]:
                    guards += f"if ({arg}.has_value()) "
                    guards += f"if ({arg}.value().scalar_type() == {atype}) "
                else:
                    guards += f"if ({arg}.scalar_type() == {atype}) "

        # Guard on constant values.
        for i, val in spec["constants"].items():
            arg = func.arg_names[i]
            if isinstance(val, bool):
                guards += f"if ({arg}) " if val else f"if (!({arg})) "
            elif isinstance(val, str):
                guards += f'if ({arg} == "{val}") '
            elif val is None:
                guards += f"if (!{arg}.has_value()) "
            else:
                guards += f"if ({arg} == {val}) "

        # Guard on special constants
        for name in AUTOTUNE_ATTRs.keys():
            if name in spec:
                guards += f"if ({name} == {spec[name]}) "

        # Guard on divisible_by_16
        config = spec["configs"][0]
        for i in config.divisible_by_16:
            arg = func.arg_names[i]
            if i in spec["signature"]:
                ttype = spec["signature"][i]
                if ttype.startswith("*"):
                    if i in spec["optional"]:
                        guards += (
                            f"if ((((uintptr_t){arg}.value().data_ptr()) % 16) == 0) "
                        )
                    else:
                        guards += f"if ((((uintptr_t){arg}.data_ptr()) % 16) == 0) "
                else:
                    guards += f"if (({arg} % 16) == 0) "
            elif i in spec["constants"]:
                assert (spec["constants"][i] % 16) == 0

        # Guard on divisible_by_8
        config = spec["configs"][0]
        for i in config.divisible_by_8:
            arg = func.arg_names[i]
            if i in spec["signature"]:
                ttype = spec["signature"][i]
                # divisible_by_8 is only applied to int
                if not ttype.startswith("*"):
                    guards += f"if (({arg} % 8) == 0) "
            elif i in spec["constants"]:
                assert (spec["constants"][i] % 8) == 0

        # Guard on equal_to_one
        for i in config.equal_to_1:
            arg = func.arg_names[i]
            if i in spec["signature"]:
                assert not spec["signature"][i].startswith("*")
                guards += f"if ({arg} == 1) "
            elif i in spec["constants"]:
                assert spec["constants"][i] == 1

        # Call the specialization.
        calls.append(f"{guards}return {kernel_name}({args});\n")
    return "".join(calls)


def gen_selector_proto(func, specs):
    params = gen_selector_params(func, specs)
    # Add Triton's default values for num warps/stages, etc
    for name, value in AUTOTUNE_ATTRs.items():
        params = params.replace(name, f"{name}={value}")
    return f"void {func.__name__}({params});"


def gen_failure_msg(func, *, signature=None, constants=None, optional=None, **kwargs):
    args = []
    for i, arg in enumerate(func.arg_names):
        if i in signature:
            if signature[i].startswith("*"):
                if i in optional:
                    args.append(
                        (
                            arg,
                            f"""({arg}.has_value() ? c10::toString({arg}.value().scalar_type()) : "nullptr")""",
                        )
                    )
                    args.append(
                        (
                            f"({arg}.has_value() ? {arg}.value().data_ptr : -1)",
                            f"({arg}.has_value() ? (uintptr_t){arg}.value().data_ptr() : -1)",
                        )
                    )
                else:
                    args.append((arg, f"{arg}.scalar_type()"))
                    args.append((f"{arg}.data_ptr", f"(uintptr_t){arg}.data_ptr()"))
            else:
                args.append((arg, arg))
        elif i in constants:
            if i in optional:
                args.append((f"{arg}.has_value()", f"{arg}.has_value()"))
            else:
                args.append((arg, arg))
    for name in AUTOTUNE_ATTRs.keys():
        args.append((f"{name}", f"{name}"))
    args.append(("cc", "cc"))
    return " << ".join([f'" {name}=" << {value}' for name, value in args])


def gen_selector(func, specs):
    params = gen_selector_params(func, specs)
    guarded_calls = gen_guarded_calls(func, specs)
    failure_msg = gen_failure_msg(func, **specs[0])
    return f"""
        void {func.__name__}({params}) {{
            auto cc = compute_capability();
            if (grid.x * grid.y * grid.z > 0) {{
                {guarded_calls}
                std::stringstream ss;
                ss << "[TritonAOT] No implementation found for {func.__name__}" << {failure_msg};
                throw c10::Error(ss.str());
            }}
        }}
    """


def gen_cpp_op_params(func, specs):
    signature = specs[0]["signature"]
    constants = specs[0]["constants"]
    optional = specs[0]["optional"]

    args = []
    for i, arg in enumerate(func.arg_names):
        if i in signature:
            if signature[i].startswith("*"):
                if i in optional:
                    args.append(f"const std::optional<at::Tensor>& {arg}")
                else:
                    args.append(f"at::Tensor& {arg}")
            elif signature[i].startswith("i"):
                args.append(f"int64_t {arg}")
            elif signature[i].startswith("f"):
                args.append(f"double {arg}")
            elif signature[i].startswith("bool"):
                args.append(f"bool {arg}")
        elif i in constants:
            if isinstance(constants[i], int):
                args.append(f"int64_t {arg}")
            elif isinstance(constants[i], str):
                args.append(f"const std::string& {arg}")
            elif isinstance(constants[i], bool):
                args.append(f"bool {arg}")
            elif constants[i] is None:
                args.append(f"const std::optional<at::Tensor>& {arg}")
            else:
                raise AssertionError(
                    f"Unsupported type {type(constants[i])} for {arg}."
                )
        else:
            raise AssertionError(f"Unspecified argument {arg}")
    for name, value in AUTOTUNE_ATTRs.items():
        args.append(f"{PY_TYPES_TO_CPP_TYPES[type(value)]} {name}")
    return ", ".join(args)


def gen_torch_op_params(func, specs, default_values):
    signature = specs[0]["signature"]
    constants = specs[0]["constants"]
    optional = specs[0]["optional"]
    args = []

    def gen_str_wrap(value):
        return f'\\"{value}\\"' if isinstance(value, str) else value

    def gen_default_str(arg):
        return (
            f" = {gen_str_wrap(default_values[arg])}" if arg in default_values else ""
        )

    for i, arg in enumerate(func.arg_names):
        t = chr(ord("a") + i)
        df_str = gen_default_str(arg)
        if i in signature:
            if signature[i].startswith("*"):
                if i in optional:
                    args.append(f"Tensor({t}!)? {arg}")
                else:
                    args.append(f"Tensor({t}!) {arg}")
            elif signature[i].startswith("i"):
                args.append(f"int {arg}{df_str}")
            elif signature[i].startswith("f"):
                args.append(f"float {arg}{df_str}")
            elif signature[i].startswith("bool"):
                args.append(f"bool {arg}{df_str}")
        elif i in constants:
            if isinstance(constants[i], int):
                args.append(f"int {arg}{df_str}")
            elif isinstance(constants[i], str):
                args.append(f"str {arg}{df_str}")
            elif isinstance(constants[i], bool):
                args.append(f"bool {arg}{df_str}")
            elif constants[i] is None:
                args.append(f"Tensor({t}!)? {arg}")
            else:
                raise AssertionError(
                    f"Unsupported type {type(constants[i])} for {arg}."
                )
        else:
            raise AssertionError(f"Unspecified argument {arg}")
    for name, value in AUTOTUNE_ATTRs.items():
        args.append(f"{type(value).__name__} {name}={value}")
    return ", ".join(args)


def gen_torch_op(func, specs, default_values):
    cpp_params = gen_cpp_op_params(func, specs)
    torch_params = gen_torch_op_params(func, specs, default_values)
    arg_names = func.arg_names
    arg_names += list(AUTOTUNE_ATTRs.keys())
    args = ", ".join(arg_names)
    return textwrap.dedent(
        f"""
        namespace {{
        triton::aot::gridDims dims_from_array(
            at::IntArrayRef grid
        ) {{
          return triton::aot::gridDims(
              grid.size() > 0 ? grid[0] : 1,
              grid.size() > 1 ? grid[1] : 1,
              grid.size() > 2 ? grid[2] : 1
          );
        }}

        void {func.__name__}_op(
            at::IntArrayRef grid,
            {cpp_params}
        ) {{
            triton::aot::{func.__name__}(
                dims_from_array(grid),
                {args}
            );
        }}

        void {func.__name__}_dummy_op(
            at::IntArrayRef grid,
            {cpp_params}
        ) {{
            // Do nothing.  The op is a dummy for model transform,
            // processing, and splitting services.
        }}
        }}

        TORCH_LIBRARY_FRAGMENT(triton, m) {{
          m.def("{func.__name__}(int[] grid, {torch_params}) -> ()");
        }}
        TORCH_LIBRARY_IMPL(triton, CUDA, m) {{
          m.impl("{func.__name__}", {func.__name__}_op);
        }}

        TORCH_LIBRARY_IMPL(triton, CPU, m) {{
          m.impl("{func.__name__}", {func.__name__}_dummy_op);
        }}

        TORCH_LIBRARY_IMPL(triton, Meta, m) {{
          m.impl("{func.__name__}", {func.__name__}_dummy_op);
        }}
        """
    )


def constexpr(s):
    expr = s[0] if isinstance(s, tuple) and len(s) > 1 else s

    if expr is None:
        return expr

    try:
        ret = int(expr)
        return ret
    except ValueError:
        pass
    try:
        ret = float(expr)
        return ret
    except ValueError:
        pass

    if isinstance(expr, bool):
        return expr
    if isinstance(expr, str) and expr not in CTYPES and not expr.startswith("*"):
        return expr
    return None


def convert_specs(base_specs, cc=None):
    specs = copy.deepcopy(base_specs)
    for spec in specs:
        if "cc" in spec:
            # Remove the cc field as a user-facing option
            del spec["cc"]

        divisible_by_16 = set()
        divisible_by_8 = set()
        equal_to_1 = set()
        none_args = set()
        optional_args = set()

        # See JITFunction._get_config() in triton/runtime/jit.py
        for i, s in enumerate(spec["signature"]):
            # handle the optional tensor case, s[2] == True means it has value, False means it doesn't have value
            if isinstance(s, tuple) and len(s) > 2:
                optional_args.add(i)
                if not s[2]:
                    none_args.add(i)
                    continue
            s = s[1] if isinstance(s, tuple) else s
            if isinstance(s, int):
                if s % 16 == 0:
                    divisible_by_16.add(i)
                if s % 8 == 0:
                    divisible_by_8.add(i)
                if s == 1:
                    equal_to_1.add(i)
            if s is None:
                none_args.add(i)

        # folded equal_to_1 and None
        ids_of_folded_args = equal_to_1 | none_args

        spec["configs"] = (
            instance_descriptor(
                divisible_by_16=divisible_by_16,
                equal_to_1=equal_to_1,
                ids_of_folded_args=ids_of_folded_args,
                divisible_by_8=divisible_by_8,
            ),
        )

        constexprs = {i: constexpr(s) for i, s in enumerate(spec["signature"])}
        spec["constants"] = {k: v for k, v in constexprs.items() if v is not None}
        for k in equal_to_1:
            spec["constants"][k] = 1
        for k in none_args:
            spec["constants"][k] = None
        spec["optional"] = optional_args
        spec["signature"] = {
            i: s[0] if isinstance(s, tuple) and len(s) > 1 else s
            for i, s in enumerate(spec["signature"])
            if i not in spec["constants"]
        }

    # By default, we will use cc = set(80) for SM80 (A100).
    # If cc is specified, we will use it to override the default.
    # Note if user explicitly specify cc, then final_specs will have duplicates
    # and will be deduplicated later when calling method dedup_specs
    cc = set(cc or [])
    if not IS_AMD:
        cc.add("80")
    final_specs = []
    for spec in specs:
        assert "cc" not in spec
        for c in cc:
            spec_cp = copy.deepcopy(spec)
            if not c.isdigit() or int(c) < 80:
                continue
            spec_cp["cc"] = int(c)
            final_specs.append(spec_cp)
    return final_specs


def key_names_and_idx(func):
    if hasattr(func, "key_idx"):
        arg_names = [func.arg_names[idx] for idx in func.key_idx]
        key_idx = func.key_idx
    else:
        arg_names = func.keys
        key_idx = [func.arg_names.index(arg) for arg in arg_names]
    return arg_names, key_idx


def gen_tuner_op(func, tuner_fallback, constants=None, **kwargs):
    arg_names, key_idx = key_names_and_idx(func)

    in_args = ", ".join(
        [
            f"{name}: {type(constants[idx]).__name__ if idx in constants else 'int'}"
            for idx, name in zip(key_idx, arg_names)
        ]
    )

    vals = []

    guard_list = []
    for key, cfg in func.cache.items():
        val = list(cfg.kwargs.values()) + [cfg.num_warps, cfg.num_stages]
        val = tuple(val)
        vals.append(val)
        equations = []
        for arg, value in zip(arg_names, key):
            if isinstance(value, str):
                equations.append(f"{arg} == '{value}'")
            elif isinstance(value, bool):
                equations.append(f"{arg} == {int(value)}")
            else:
                equations.append(f"{arg} == {value}")
        guard_list.append(f"if {' and '.join(equations)}: return {val}")
    guards = "\n        ".join(guard_list)

    name = unwrap_heuristic(func).__name__
    meta = name + "_meta"

    fmt_args = ", ".join([f"{{{arg_name}}}" for arg_name in arg_names])

    raise_runtime_error_str = (
        f"""raise RuntimeError(f"No autotuning config found for {name}({fmt_args})")"""
    )
    fallback_str = f"""return {Counter(vals).most_common(1)[0][0]}"""

    return textwrap.dedent(
        f"""
    def {meta}({in_args}):
        {guards}
        {fallback_str if tuner_fallback else raise_runtime_error_str}
    """
    )


def gen_tuner_op_cpp(func, tuner_fallback, constants=None, **kwargs):
    def infer_arg_type(idx):
        if idx in constants:
            return PY_TYPES_TO_CPP_TYPES[type(constants[idx])]
        else:
            return "int64_t"

    arg_names, key_idx = key_names_and_idx(func)

    in_args = ", ".join(
        [f"{infer_arg_type(idx)} {name}" for idx, name in zip(key_idx, arg_names)]
    )

    vals = []
    guard_list = []
    for key, cfg in func.cache.items():
        val = list(cfg.kwargs.values()) + [cfg.num_warps, cfg.num_stages]
        val = tuple(val)
        vals.append(val)
        equations = []
        for arg, value in zip(arg_names, key):
            if isinstance(value, str):
                equations.append(f'{arg} == "{value}"')
            elif isinstance(value, bool):
                equations.append(f"{arg} == {int(value)}")
            else:
                equations.append(f"{arg} == {value}")
        guard_list.append(f"if ({' && '.join(equations)}) return std::make_tuple{val};")
    guards = "\n        ".join(guard_list)
    name = unwrap_heuristic(func).__name__
    meta = name + "_meta"
    fmt_args = ", ".join([f"{arg_name}" for arg_name in arg_names])
    raise_runtime_error_str = f"""throw std::runtime_error("No autotuning config found for {name}({fmt_args})");"""
    fallback_str = f"""return std::make_tuple{Counter(vals).most_common(1)[0][0]};"""
    # Infer the return type from the actual values
    return_type = infer_return_type(vals[0])
    return textwrap.dedent(
        f"""
    inline std::tuple<{return_type}> {meta}({in_args}) {{
        {guards}
        {fallback_str if tuner_fallback else raise_runtime_error_str}
    }}
    """
    )


def infer_return_type(vals) -> str:
    # This function should infer the return type based on the actual values in `vals`
    types = [PY_TYPES_TO_CPP_TYPES.get(type(val)) for val in vals]
    try:
        # pyre-fixme[6]: For 1st argument expected
        #  `Iterable[typing_extensions.LiteralString]` but got `List[Optional[str]]`.
        return ", ".join(types)
    except TypeError:  # one of the types cannot be inferred, e.g. `None`
        raise ValueError("Cannot infer return type from `vals`")


def autotune_specs(func, specs):
    tuned_specs = []
    for spec in specs:
        for cfg in func.cache.values():
            constants = spec.get("constants", {}).copy()
            for arg_name, arg_val in cfg.kwargs.items():
                if arg_name in AUTOTUNE_ATTRs:
                    continue
                arg_idx = func.arg_names.index(arg_name)
                if constants.get(arg_idx, -1) == -1:
                    constants[arg_idx] = arg_val
            base_spec = {
                "cc": spec["cc"],
                "signature": spec["signature"],
                "constants": constants,
                "configs": spec["configs"],
                "optional": spec["optional"],
            }
            for name, value in AUTOTUNE_ATTRs.items():
                if name in cfg.kwargs:
                    base_spec[name] = cfg.kwargs[name]
                else:
                    base_spec[name] = getattr(cfg, name, value)
                # AMD has changed their software pipeliner in Triton
                # It now expects num_stages == 2 instead of 0
                # see: https://github.com/pytorch/pytorch/pull/139881
                # if we see someone try to set num_stages == 0, set it to the default (2) instead
                # We can't use the Triton hook to get the default value because it requires the AMD runtime to be loaded
                if (
                    IS_AMD
                    and name == "num_stages"
                    and base_spec[name] == 0
                    and TRITON_VERSION >= "3.2.0"
                ):
                    base_spec[name] = 2

            tuned_specs.append(base_spec)
    return tuned_specs


def gen_compile_arg(
    spec: dict[str, Any], func: JITFunction, attrs: instance_descriptor | None = None
) -> tuple[ASTSource]:
    signature = spec["signature"]
    constants = spec.get("constants", {})
    # Upstream https://github.com/triton-lang/triton/commit/9743ec0dca5bbd9dbce20adc3ee273af6b095f94
    # removed AttrsDescriptor and used a plain dict instead.
    if AttrsDescriptor is None:
        signature = {
            list(func.signature.parameters.keys())[k]: v for k, v in signature.items()
        }
        constants = {
            list(func.signature.parameters.keys())[k]: v for k, v in constants.items()
        }
        # tl.constexprs in the kernel signature need to be added to the
        # signature passed to ASTSource also. e.g.
        #     SILU_U: tl.constexpr,
        #     BLOCK_D: tl.constexpr,
        #     TRAINING: tl.constexpr,
        #     CONCAT_UX: tl.constexpr,
        for k in constants:
            signature[k] = "constexpr"
        compile_arg = (
            # pyre-ignore[28]: Unexpected keyword argument `constexprs` to ...
            ASTSource(
                func,
                signature,
                constexprs=constants,
                attrs=attrs._asdict() if attrs is not None else None,
            ),
        )
    # Pre-9743ec0dca5bbd9dbce20adc3ee273af6b095f94 API
    else:
        if attrs is None:
            pass
        elif hasattr(AttrsDescriptor, "ids_of_folded_args"):
            attrs = AttrsDescriptor(*attrs)
        elif hasattr(AttrsDescriptor, "arg_properties"):
            attrs = AttrsDescriptor.from_dict(
                {
                    "cls": "AttrsDescriptor",
                    "arg_properties": {
                        "tt.divisibility": attrs.divisible_by_16,
                        "tt.equal_to": attrs.equal_to_1,
                    },
                }
            )
            signature = {
                list(func.signature.parameters.keys())[k]: v
                for k, v in signature.items()
            }
            constants = {
                list(func.signature.parameters.keys())[k]: v
                for k, v in constants.items()
            }
        else:
            attrs = AttrsDescriptor(
                divisible_by_16=attrs.divisible_by_16,
                equal_to_1=attrs.equal_to_1,
            )
        compile_arg = (
            # pyre-ignore[28]: Unexpected keyword argument `constants` to ...
            ASTSource(
                func,
                signature,
                constants=constants,
                attrs=attrs,
            ),
        )
    return compile_arg


# For each spec, generate a kernel:
# - cubin
# - loader
# - launcher
def spec_gen(args):
    install_dir, spec, module, name, objcopy = args

    # To run this function with multiprocessing, we need to import the function by name,
    # since JITFunction cannot be pickled.
    # we have the case where the func name is injected with a suffix, like "_cuda" or "_amd",
    # we should use the original name to import the func in such case
    original_name = name
    splits = name.split("_")
    end_idx = len(splits)
    while end_idx > 0:
        original_name = "_".join(splits[:end_idx])
        if hasattr(importlib.import_module(module), original_name):
            break
        end_idx -= 1
    func = unwrap_heuristic(getattr(importlib.import_module(module), original_name))
    func.__name__ = name

    # Generate cubin.
    kernel_name = gen_kernel_name(func, **spec)

    attrs = (
        spec["configs"][0] if "configs" in spec and len(spec["configs"]) > 0 else None
    )

    compile_arg = gen_compile_arg(spec, func, attrs)
    # TODO(lufang): fix spec["cc"] to use full format instead of digit only.
    if IS_AMD:
        target = [
            "hip",
            HIP_CC_TO_ARCH_INFO[spec["cc"]],
            get_warp_size(),
        ]
    else:
        target = ["cuda", spec["cc"], get_warp_size()]
    if hasattr(triton.backends.compiler, "GPUTarget"):
        target = triton.backends.compiler.GPUTarget(*target)

    options = {name: spec[name] for name in AUTOTUNE_ATTRs.keys()}
    compile_kwargs = {
        "target": target,
        "options": options,
    }
    kernel = triton.compiler.compile(*compile_arg, **compile_kwargs)
    metadata_name = kernel.metadata.name
    metadata_shared = kernel.metadata.shared
    cubin = gen_cubin(kernel_name, kernel, install_dir, objcopy=objcopy)
    out = [
        cubin,
        # Generate loader.
        gen_loader(kernel_name, metadata_name, metadata_shared),
        # Generate launcher.
        gen_launcher(kernel_name, func, kernel, metadata_shared, **spec),
    ]
    return "".join(out)


def add_defaults(specs):
    specs = list(specs)
    for spec in specs:
        for name, value in AUTOTUNE_ATTRs.items():
            if name not in spec:
                spec[name] = value
    return specs


def gen_loader_helper() -> str:
    result = textwrap.dedent(
        """
        namespace {
        #ifdef USE_ROCM
        void check_errors(int shared, CUfunction func) {{
            return;
        }}
        #else
        void check_errors(int shared, CUfunction func) {{
            int shared_optin;
            int device = 0;
            CUDA_CHECK(cuDeviceGetAttribute(
                &shared_optin, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
                device));
            if (shared > 49152 && shared_optin > 49152) {{
              CUDA_CHECK(cuFuncSetCacheConfig(func, CU_FUNC_CACHE_PREFER_SHARED));
              int shared_total, shared_static;
               CUDA_CHECK(cuDeviceGetAttribute(
                    &shared_total, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR,
                    device));
                CUDA_CHECK(cuFuncGetAttribute(&shared_static,
                                              CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, func));
                CUDA_CHECK(
                    cuFuncSetAttribute(func, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                       shared_optin - shared_static));
            }}
        }}
        #endif
        } // namespace
        """
    )
    return result


def sigchld_handler(signum, frame):
    sketchy_signals = map(int, [signal.SIGSEGV, signal.SIGABRT, signal.SIGBUS])
    try:
        # Consume all pending SIGCHLDs, looking for unexpected failures
        while True:
            pid, status = os.waitpid(-1, os.WNOHANG)
            if pid == 0:
                break
            if os.WIFSIGNALED(status) and os.WTERMSIG(status) in sketchy_signals:
                logger.error(
                    f"Child process {pid} exited catastrophically with signal {os.WTERMSIG(status)}, terminating!"
                )

                # Avoid triggering atexit etc which can get stuck and behave improperly
                # because multiprocessing sets up an atexit handler to join workers
                # (sigh).  We want to exit, now, so use os._exit instead of sys.exit.
                os._exit(1)
    except ChildProcessError:
        pass


def dedup_specs(specs):
    # Dedup specs to avoid duplicate compilation.
    deduped_specs = []
    duplicated_specs = []
    hash_spec_ids = set()
    for spec in specs:
        id = hash_spec(spec)
        if id in hash_spec_ids:
            duplicated_specs.append(spec)
        else:
            hash_spec_ids.add(id)
            deduped_specs.append(spec)

    logger.debug(
        f"[TritonAOT Dedup] {len(specs)=} {len(deduped_specs)=} {len(duplicated_specs)=}"
    )
    return deduped_specs


def compile(
    func,
    base_specs,
    install_dir,
    prefix,
    *,
    default_values=None,
    objcopy=None,
    ld=None,
    ar=None,
    tuner_fallback=False,
    cc=None,
):
    specs = convert_specs(base_specs, cc)
    tuned_func = None
    default_values = {} if default_values is None else default_values

    # Python's multiprocessing.Pool class is not great at handling unexpected child
    # failures such as segfaults.  Account for this by temporarily installing a signal
    # handler that considers such signals a catastrophic compilation failure.  If not
    # for this, the Pool will deadlock.
    previous_child_handler = signal.signal(signal.SIGCHLD, sigchld_handler)

    if isinstance(func, triton.runtime.autotuner.Autotuner):
        specs = autotune_specs(func, specs)
        tuned_func = func

    func = unwrap_heuristic(func)

    # sanity check to make sure args with default values are always at the end
    has_default_value_arg = False
    for name in func.arg_names:
        if name in default_values:
            has_default_value_arg = True
        elif has_default_value_arg:
            raise RuntimeError(
                f"default values must be at the end of the argument list. {func.arg_names=} {default_values=}"
            )

    specs = add_defaults(specs)
    specs = dedup_specs(specs)

    h_out = f"{install_dir}/{prefix}.h"
    cu_out = f"{install_dir}/{prefix}.cpp"
    torch_out = f"{install_dir}/{prefix}_torch_op.cpp"
    py_out = f"{install_dir}/{prefix}.py"
    cubin_out = f"{install_dir}/{prefix}.o"
    lib_a_out = f"{install_dir}/{prefix}.a"

    with open(h_out, "w") as fp:
        fp.write("#pragma once\n\n")
        fp.write(
            textwrap.dedent(
                """
            #include <ATen/Tensor.h>
            #include <cuda.h>
            #include <cuda_runtime.h>
            #include "torch/types.h"

            namespace triton {
            namespace aot {

        """
            )
        )
        fp.write(
            textwrap.dedent(
                """
            #ifndef GRID_DIM_DEFINED_MACRO
            struct gridDims {
              int x = 1;
              int y = 1;
              int z = 1;
              cudaStream_t stream = 0;
              gridDims(int _x = 1, int _y = 1, int _z = 1, cudaStream_t _stream = 0)
                : x(_x), y(_y), z(_z), stream(_stream) {}
            };
            #define GRID_DIM_DEFINED_MACRO
            #endif
        """
            )
        )
        if tuned_func:
            fp.write(gen_tuner_op_cpp(tuned_func, tuner_fallback, **specs[0]))
        fp.write(gen_selector_proto(func, specs))
        fp.write("\n")
        fp.write(
            textwrap.dedent(
                """
            } // namespace aot
            } // namespace triton

        """
            )
        )

    with open(cu_out, "w") as fp:
        fp.write(
            textwrap.dedent(
                f"""
            #include "{prefix}.h"
            #include <ATen/Tensor.h>
            #include <ATen/core/op_registration/op_registration.h>
            #include <ATen/cuda/CUDAContext.h>
            #include <ATen/cuda/Exceptions.h>
            #include <ATen/cuda/nvrtc_stub/ATenNVRTC.h>
            #include <c10/cuda/CUDAStream.h>
            #include <c10/util/Exception.h>
            #include <torch/library.h>

            #define CUDA_CHECK(ans) AT_CUDA_DRIVER_CHECK(ans)

            namespace triton {{
            namespace aot {{

            namespace {{
            int compute_capability() {{
                auto major = at::cuda::getCurrentDeviceProperties()->major;
                auto minor = at::cuda::getCurrentDeviceProperties()->minor;
                return major * 10 + minor;
            }}
            }} // namespace
        """
            )
        )

        check_errors_string = gen_loader_helper()
        fp.write(check_errors_string)

        if mp.get_start_method(allow_none=True) is None:
            mp.set_start_method("spawn")
        max_procs = mp.cpu_count() // 2 + 1
        with mp.Pool(min(len(specs), max_procs)) as pool:
            outputs = pool.map(
                spec_gen,
                [
                    (
                        install_dir,
                        spec,
                        func.__module__,
                        func.__name__,
                        objcopy,
                    )
                    for spec in specs
                ],
            )

        generated_specs = "\n".join(outputs)

        fp.write(generated_specs)
        fp.write(gen_selector(func, specs))
        fp.write(
            textwrap.dedent(
                """
            } // namespace aot
            } // namespace triton
        """
            )
        )

    with open(torch_out, "w") as fp:
        fp.write(
            textwrap.dedent(
                f"""
            #include "{prefix}.h"
            #include <ATen/Tensor.h>
            #include <torch/library.h>
        """
            )
        )
        fp.write(gen_torch_op(func, specs, default_values))

    o_file_suffix = "*.hsaco.o" if IS_AMD else "*.cubin.o"

    if objcopy:
        assert ld is not None
        subprocess.run(
            shlex.split(ld)
            + ["--relocatable", "-o", cubin_out]
            + sorted(glob.glob(f"{install_dir}/{o_file_suffix}"))
        )
        assert ar is not None
        subprocess.run(shlex.split(ar) + ["rcsD", lib_a_out, cubin_out])

    if tuned_func:
        with open(py_out, "w") as fp:
            fp.write(gen_tuner_op(tuned_func, tuner_fallback, **specs[0]))

    signal.signal(signal.SIGCHLD, previous_child_handler)
