"""Shared helpers for op verification scripts (verify_cpu.py / verify_npu.py).

Self-contained: only depends on the op asset files themselves
({op}.py + {op}.json), no external dataset needed.
"""

import importlib.util
import json
import os

import torch


def load_op_module(py_path, tag=None):
    """Load an op reference-implementation module from its .py path."""
    name = "_opverify_%s" % (tag or os.path.basename(py_path)[:-3])
    spec = importlib.util.spec_from_file_location(name, py_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def count_json_cases(json_path):
    with open(json_path, "r") as f:
        return sum(1 for line in f if line.strip())


def clone_arg(a):
    if isinstance(a, torch.Tensor):
        return a.clone()
    if isinstance(a, list):
        return [clone_arg(x) for x in a]
    if isinstance(a, tuple):
        return tuple(clone_arg(x) for x in a)
    return a


def run_forward(Model, init, args):
    """Run Model(*init)(*args): no_grad first, grad-path fallback for *Grad ops.
    Inputs are cloned first -- several reference Models mutate inputs in place.
    Returns None on success, the exception on failure.
    """
    args = [clone_arg(a) for a in args]
    init = [clone_arg(a) for a in init]

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
        return _go(False), None
    except Exception:
        try:
            return _go(True), None
        except Exception as e:
            return None, e


def to_device(a, device):
    if isinstance(a, torch.Tensor):
        return a.to(device)
    if isinstance(a, list):
        return [to_device(x, device) for x in a]
    if isinstance(a, tuple):
        return tuple(to_device(x, device) for x in a)
    return a


def nonfinite_count(out):
    """Count non-finite elements across output tensors (inf/nan).
    Robust to dtypes lacking an isfinite kernel on the current device
    (e.g. aclnnIsFinite has no complex64 support on NPU): complex tensors are
    checked via real/imag parts; any remaining kernel failure skips the scan."""
    n = 0
    outs = out if isinstance(out, (tuple, list)) else [out]
    for o in outs:
        if not isinstance(o, torch.Tensor):
            continue
        if not (o.is_floating_point() or o.is_complex()):
            continue
        try:
            if o.is_complex():
                n += int((~torch.isfinite(o.real)).sum()) + int((~torch.isfinite(o.imag)).sum())
            else:
                n += int((~torch.isfinite(o)).sum())
        except Exception:
            pass
    return n


def run_op(py_path, device=None):
    """Run all cases of one op. Returns a result dict:
    {op, total, passed, warns, fails}
    device=None -> CPU; device='npu' -> move inputs to NPU first.
    """
    base = os.path.basename(py_path)[:-3]
    json_path = py_path[:-3] + ".json"
    res = {"op": base, "total": 0, "passed": 0, "warns": [], "fails": []}
    try:
        n_json = count_json_cases(json_path)
        mod = load_op_module(py_path)
        groups = mod.get_input_groups()
        inits = mod.get_init_inputs()
        res["total"] = len(groups)
        if len(groups) != n_json:
            res["fails"].append("case count %d != json lines %d" % (len(groups), n_json))
            return res
        for c, args in enumerate(groups):
            init = inits[c] if inits else []
            if device is not None:
                args = to_device(args, device)
                init = to_device(init, device)
            try:
                out, err = run_forward(mod.Model, init, args)
            except Exception as e:
                err = e
                out = None
            if err is not None:
                res["fails"].append("case %d: %s" % (c, repr(err)[:200]))
                if len(res["fails"]) >= 5:
                    break
                continue
            try:
                nf = nonfinite_count(out)
            except Exception:
                nf = 0  # scan failure must not fail the case
            if nf:
                res["warns"].append("case %d: %d non-finite output elements" % (c, nf))
            res["passed"] += 1
    except Exception as e:
        res["fails"].append("harness: " + repr(e)[:300])
    return res


def write_report(results, path, title, strict_finite=False):
    """Markdown report; returns (n_ops_ok, n_warn_ops, n_fail_ops)."""
    results = sorted(results, key=lambda r: r["op"])
    fails = [r for r in results if r["passed"] != r["total"] or r["fails"]]
    warns = [r for r in results if r["warns"]]
    if strict_finite:
        fails = [r for r in results if r["passed"] != r["total"] or r["fails"] or r["warns"]]
    total_cases = sum(r["total"] for r in results)
    passed_cases = sum(r["passed"] for r in results)
    lines = ["# %s" % title, "",
             "| 项目 | 数量 |", "|---|---|",
             "| 算子总数 | %d |" % len(results),
             "| 全部通过的算子 | %d |" % (len(results) - len(fails)),
             "| 失败算子 | %d |" % len(fails),
             "| 有限性 WARN 算子 | %d |" % len(warns),
             "| 用例通过率 | %d/%d |" % (passed_cases, total_cases), ""]
    if fails:
        lines += ["## 失败算子", "", "| 算子 | 通过/总数 | 首个错误 |", "|---|---|---|"]
        for r in fails:
            first = (r["fails"][0] if r["fails"] else (r["warns"][0] if r["warns"] else ""))
            lines.append("| %s | %d/%d | %s |" % (r["op"], r["passed"], r["total"],
                                                  first.replace("|", "\\|")[:150]))
        lines.append("")
    if warns:
        lines += ["## 有限性 WARN（不判失败，仅记录）", "",
                  "| 算子 | 详情 |", "|---|---|"]
        for r in warns:
            detail = "; ".join(r["warns"][:3]).replace("|", "\\|")[:150]
            lines.append("| %s | %s |" % (r["op"], detail))
        lines.append("")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines))
    return len(results) - len(fails), len(warns), len(fails)
