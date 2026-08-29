#!/usr/bin/env python3
"""Process a batch of CANN_Ops operators on NPU in one long-lived process
(amortizes torch_npu/ACL init). Reads op names from argv, appends one JSON
line per op to the result file given by --out.

Usage: npu_batch.py --out /tmp/npu_chunk_N.jsonl cannops_... cannops_...
ASCEND_RT_VISIBLE_DEVICES selects the card (set by the orchestrator).
"""

import importlib.util
import gc
import json
import os
import sys
import traceback

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "CANN_Ops", "src")


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


def run_op(base):
    import torch
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
        if len(fails) >= 100:
            break
    total = len(groups)
    del mod, groups, inits
    return {"op": base, "total": total, "passed": total - len(fails), "fails": fails}


def main():
    args = sys.argv[1:]
    out_path = args[args.index("--out") + 1]
    bases = [a for a in args if not a.startswith("--") and a != out_path]
    import torch
    import torch_npu  # noqa
    with open(out_path, "a") as out:
        for base in bases:
            try:
                r = run_op(base)
            except Exception:
                r = {"op": base, "total": -1, "passed": 0,
                     "fails": ["op crash: " + traceback.format_exc(limit=2)[-300:]]}
            out.write(json.dumps(r, ensure_ascii=False) + "\n")
            out.flush()
            gc.collect()
            try:
                torch.npu.empty_cache()
            except Exception:
                pass
            print("[%d/%d] %s" % (r["passed"], r["total"], base), flush=True)


if __name__ == "__main__":
    main()
