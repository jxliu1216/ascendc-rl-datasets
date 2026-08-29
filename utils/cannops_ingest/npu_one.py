#!/usr/bin/env python3
"""Run ONE CANN_Ops operator's reference Model on NPU for all its cases.
Prints a single JSON line: {"op": ..., "total": N, "passed": M, "fails": [...]}

Usage: npu_one.py <new_base>   (e.g. cannops_level1_0_AbsMath)
ASCEND_RT_VISIBLE_DEVICES selects the card (set by the orchestrator).
"""

import importlib.util
import json
import os
import sys
import traceback

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "CANN_Ops")


def to_npu(a):
    import torch
    if isinstance(a, torch.Tensor):
        return a.npu()
    if isinstance(a, list):
        return [to_npu(x) for x in a]
    if isinstance(a, tuple):
        return tuple(to_npu(x) for x in a)
    return a


def run_forward(Model, init, args):
    import torch
    def _go(grad):
        a2 = []
        for a in args:
            if grad and isinstance(a, torch.Tensor) and (a.is_floating_point() or a.is_complex()):
                a = a.clone().requires_grad_(True)
            a2.append(a)
        model = Model(*init)
        if grad:
            out = model(*a2)
            outs = out if isinstance(out, (tuple, list)) else [out]
            for o in outs:
                if isinstance(o, torch.Tensor) and (o.is_floating_point() or o.is_complex()):
                    if o.grad_fn is not None:
                        o.sum().backward(retain_graph=True)
            return out
        with torch.no_grad():
            return model(*a2)
    try:
        _go(False)
        return None
    except Exception:
        try:
            _go(True)
            return None
        except Exception as e:
            return e


def main():
    base = sys.argv[1]
    import torch  # noqa
    import torch_npu  # noqa

    spec = importlib.util.spec_from_file_location(base, os.path.join(OUT_DIR, base + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    groups = mod.get_input_groups()
    inits = mod.get_init_inputs()
    fails = []
    for c, args in enumerate(groups):
        init = inits[c] if inits else []
        try:
            err = run_forward(mod.Model, to_npu(init), to_npu(args))
        except Exception as e:
            err = e
        if err is not None:
            fails.append("case %d: %s" % (c, repr(err)[:200]))
        if len(fails) >= 5:
            break
    print(json.dumps({"op": base, "total": len(groups),
                      "passed": len(groups) - len(fails), "fails": fails},
                     ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print(json.dumps({"op": sys.argv[1] if len(sys.argv) > 1 else "?",
                          "total": 0, "passed": 0,
                          "fails": ["harness crash: " + traceback.format_exc(limit=2)[-300:]]},
                         ensure_ascii=False))
