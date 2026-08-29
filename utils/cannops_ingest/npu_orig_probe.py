#!/usr/bin/env python3
"""Probe: run the ORIGINAL cann_ops_tmp op on NPU under two variants:
  A) as-is (torch_npu stubbed by the file itself, factories unpatched)
  B) source-repo validation conditions: real torch_npu + transfer_to_npu
     imported first (stubs skipped, factories default to NPU)
Prints one JSON line: {"op":..., "A": [...case errs...], "B": [...], "total": N}

Usage: npu_orig_probe.py <level> <old_id> <opname>
"""

import importlib.util
import json
import os
import sys

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "cann_ops_tmp")


def to_npu(a):
    import torch
    if isinstance(a, torch.Tensor):
        return a.npu() if a.device.type == "cpu" else a
    if isinstance(a, list):
        return [to_npu(x) for x in a]
    if isinstance(a, tuple):
        return tuple(to_npu(x) for x in a)
    return a


def load_orig(level, old_id, op):
    path = os.path.join(SRC, level, "%d_%s.py" % (old_id, op))
    spec = importlib.util.spec_from_file_location("orig_%s" % op, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # json on disk is {oldid}_{Op}.json while the file hardcodes {Op}.json
    json_path = os.path.join(SRC, level, "%d_%s.json" % (old_id, op))

    def _patched(json_path=json_path):
        with open(json_path, "r") as f:
            return [json.loads(line) for line in f if line.strip()]

    if hasattr(mod, "_load_cases"):
        mod._load_cases = _patched
    import torch
    for fn_name in ("get_inputs", "get_init_inputs_per_case"):
        if hasattr(mod, fn_name):
            _orig_fn = getattr(mod, fn_name)

            def _cpu_fn(param, device=None, _orig_fn=_orig_fn):
                return _orig_fn(param, device=torch.device("cpu"))

            setattr(mod, fn_name, _cpu_fn)
    return mod


def run_all(mod, move):
    groups = mod.get_input_groups()
    inits = mod.get_init_inputs()
    errs = []
    import torch
    for c, args in enumerate(groups):
        init = inits[c] if inits else []
        try:
            if move:
                args, init = to_npu(args), to_npu(init)
            model = mod.Model(*init)
            with torch.no_grad():
                model(*args)
            errs.append(None)
        except Exception as e:
            errs.append(repr(e)[:150])
    return errs


def main():
    level, old_id, op = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    # torch must be imported (backend autoload resolved) before loading the
    # original file, whose torch_npu stub would otherwise break the autoload
    import torch  # noqa
    import torch_npu  # noqa
    out = {"op": op, "total": None, "A": None, "B": None}
    # Variant A: file as-is (its own stub of torch_npu)
    mod = load_orig(level, old_id, op)
    errs = run_all(mod, move=True)
    out["A"] = [i for i, e in enumerate(errs) if e]
    out["A_err0"] = next((e for e in errs if e), None)
    out["total"] = len(errs)
    del mod
    # Variant B: real torch_npu + transfer_to_npu first (source validation cond.)
    try:
        import torch  # noqa
        import torch_npu  # noqa
        from torch_npu.contrib import transfer_to_npu  # noqa
        mod = load_orig(level, old_id, op)
        errs = run_all(mod, move=True)
        out["B"] = [i for i, e in enumerate(errs) if e]
        out["B_err0"] = next((e for e in errs if e), None)
    except Exception as e:
        out["B_crash"] = repr(e)[:300]
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
