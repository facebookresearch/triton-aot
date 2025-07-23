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

import os
import pickle

from inspect import getcallargs, Parameter, signature

from typing import Any, Dict, List

import torch

# @manual=//triton:triton
import triton.language as tl

# @manual=//triton:triton
from triton.runtime.autotuner import Autotuner

# @manual=//triton:triton
from triton.runtime.jit import KernelInterface

from triton_aot.compiler import compile, hash_spec, unwrap_heuristic

try:
    from triton_aot.fb.utils import build_shared_lib
except ImportError:
    from triton_aot.utils import build_shared_lib

TRITON_AOT_KERNEL_SPECS = {}
TRITON_AOT_SPECS_HASHSET = {}
ENABLE_TRITON_AOT_COMPILE = False


def _unwrap_triton_fn(fn):
    while isinstance(fn, KernelInterface):
        fn = fn.fn
    return fn


def _infer_spec(fn, scalar_annotations, *args, **kwargs) -> Dict[str, List[Any]]:  # noqa: C901
    triton_fn = _unwrap_triton_fn(fn)
    fn_sig = signature(triton_fn)
    arg_annotations = {}
    arg_default_values = {}
    for arg_name, param in fn_sig.parameters.items():
        arg_annotations[arg_name] = param.annotation
        arg_default_values[arg_name] = param.default
    if isinstance(fn, Autotuner):
        auto_tune_args = fn.configs[0].kwargs.keys()
    else:
        auto_tune_args = []
    spec = []
    clean_kwargs = {k: v for k, v in kwargs.items() if k != "warmup" and k != "grid"}
    for arg_name in auto_tune_args:
        if arg_name not in clean_kwargs:
            clean_kwargs[arg_name] = -1
    call_args = getcallargs(triton_fn, *args, **clean_kwargs)

    for arg_name in fn_sig.parameters.keys():
        arg = call_args[arg_name]
        if arg_annotations[arg_name] != Parameter.empty:
            if arg_annotations[arg_name] == tl.constexpr:
                spec.append(arg)
            else:
                RuntimeError(
                    f"TritonAOT: unsupported scalar annotation {arg_annotations[arg_name]}."
                )
        elif arg_name in scalar_annotations:
            spec.append(scalar_annotations[arg_name])
        else:
            if isinstance(arg, torch.Tensor):
                if arg.dtype == torch.float64:
                    spec.append(("*fp64", 16))
                elif arg.dtype == torch.float32:
                    spec.append(("*fp32", 16))
                elif arg.dtype == torch.float16:
                    spec.append(("*fp16", 16))
                elif arg.dtype == torch.bfloat16:
                    spec.append(("*bf16", 16))
                else:
                    raise RuntimeError(
                        f"TritonAOT: unsupport tensor type: str{arg.dtype}."
                    )
            elif isinstance(arg, int):
                spec.append("i64")
            elif isinstance(arg, float):
                spec.append("fp32")
            elif arg is None:
                spec.append(None)
            else:
                raise RuntimeError(f"TritonAOT: parameter {arg_name} needs annotation.")
    return {"signature": spec}


class TritonAOT(KernelInterface):
    def __init__(self, fn, annotations) -> None:
        self.fn = fn
        self.annotations = annotations

    def run(self, *args, **kwargs):
        if ENABLE_TRITON_AOT_COMPILE:
            global TRITON_AOT_KERNEL_SPECS
            global TRITON_AOT_KERNEL_SPECS_HASHSET
            spec = _infer_spec(self.fn, self.annotations, *args, **kwargs)
            if self.fn not in TRITON_AOT_KERNEL_SPECS:
                TRITON_AOT_KERNEL_SPECS[self.fn] = {}
                TRITON_AOT_KERNEL_SPECS[self.fn] = []
                TRITON_AOT_KERNEL_SPECS_HASHSET[self.fn] = set()
            hashed_spec = hash_spec(spec)
            if hashed_spec not in TRITON_AOT_KERNEL_SPECS_HASHSET[self.fn]:
                TRITON_AOT_KERNEL_SPECS[self.fn].append(spec)
                TRITON_AOT_KERNEL_SPECS_HASHSET[self.fn].add(hashed_spec)
            ret = self.fn.run(*args, **kwargs)
        else:
            ret = self.fn.run(*args, **kwargs)
        return ret


def triton_aot(annotations):
    def decorator(fn):
        return TritonAOT(fn, annotations)

    return decorator


class triton_aot_compile:
    def __init__(self):
        pass

    def __enter__(self):
        global TRITON_AOT_KERNEL_SPECS
        global TRITON_AOT_KERNEL_SPECS_HASHSET
        global ENABLE_TRITON_AOT_COMPILE

        TRITON_AOT_KERNEL_SPECS = {}
        TRITON_AOT_KERNEL_SPECS_HASHSET = {}
        ENABLE_TRITON_AOT_COMPILE = True

    def __exit__(self, exc_type, exc_value, traceback):
        global TRITON_AOT_KERNEL_SPECS
        global ENABLE_TRITON_AOT_COMPILE

        triton_aot_dir = os.getenv("TRITON_AOT_DIR", default=".triton_aot")
        if not os.path.exists(triton_aot_dir):
            os.makedirs(triton_aot_dir)
        cc = os.getenv("TRITON_AOT_ARCH", default="80")
        ld = os.getenv("TRITON_AOT_LD_PATH", default="ld")
        objcopy = os.getenv("TRITON_AOT_OBJCOPY_PATH", default="objcopy")
        ar = os.getenv("TRITON_AOT_AR_PATH", default="ar")

        for fn, specs in TRITON_AOT_KERNEL_SPECS.items():
            jit_fn = unwrap_heuristic(fn)
            fn_name = jit_fn.__name__
            fn_dir = (
                f"{triton_aot_dir}/{jit_fn.__module__.rsplit('.', 1)[-1]}_{fn_name}"
            )
            if not os.path.exists(fn_dir):
                os.makedirs(fn_dir)
            with open(f"{fn_dir}/{fn_name}_autotune_cache", "wb") as data:
                pickle.dump(fn.cache, data)  # noqa

            compile(
                func=fn,
                base_specs=specs,
                install_dir=f"{fn_dir}",
                prefix=f"{fn_name}",
                objcopy=objcopy,
                ld=ld,
                ar=ar,
                cc=cc,
                tuner_fallback=True,
            )

            build_shared_lib(fn_dir=fn_dir, fn_name=fn_name)

        ENABLE_TRITON_AOT_COMPILE = False
