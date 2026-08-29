#!/usr/bin/env python3
"""Verify CANN_Ops/ converted operators against cann_ops_tmp originals.

Per operator, three checks:
1. structural  -- new get_input_groups()/get_init_inputs() vs original:
                  case count, arg count, tensor dtype/shape, attr values (exact)
2. forward     -- every case runs Model.forward on CPU with the NEW inputs
                  (no_grad first, grad-path fallback); outputs' finite pattern
                  must match the original inputs' outputs
3. distribution-- pooled per-arg stats (min/max/mean/std, true-fraction for
                  bool, unique-ratio for int) of new vs original generator;
                  mismatch => REVIEW (not a hard failure)

Writes CANN_Ops/_ingest_report.md and prints a summary.
Exit code 0 iff every op is PASS or REVIEW (no FAIL).
"""

import gc
import importlib.util
import json
import os
import sys
import traceback

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(HERE, "..", "..", "cann_ops_tmp")
OUT_DIR = os.path.join(HERE, "..", "..", "CANN_Ops", "src")
REPORT = os.path.join(HERE, "..", "..", "CANN_Ops", "report", "_ingest_report.md")
SAMPLES = 3
STAT_RTOL = 0.25
STAT_ATOL = 0.15


_TRUE_RANDN, _TRUE_RAND = torch.randn, torch.rand
_SAFE_INT_DTYPES = {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8, torch.bool}


def _safe_make(_orig, _lo, _hi, *args, **kwargs):
    dtype = kwargs.get("dtype")
    if dtype in _SAFE_INT_DTYPES:
        size = args
        if len(args) == 1 and isinstance(args[0], (list, tuple)):
            size = tuple(args[0])
        new_kwargs = dict(kwargs)
        if dtype in (torch.uint8, torch.bool):
            _lo, _hi = 0, (2 if dtype == torch.bool else 100)
        return torch.randint(_lo, _hi, tuple(int(x) for x in size), **new_kwargs)
    return _orig(*args, **kwargs)


def load_module(path, tag, json_name=None):
    spec = importlib.util.spec_from_file_location("_verify_%s" % tag, path)
    mod = importlib.util.module_from_spec(spec)
    # source files re-wrap torch.randn/rand at import; keep chain depth at 1
    torch.randn, torch.rand = _TRUE_RANDN, _TRUE_RAND
    spec.loader.exec_module(mod)
    torch.randn = lambda *a, **kw: _safe_make(_TRUE_RANDN, -100, 100, *a, **kw)
    torch.rand = lambda *a, **kw: _safe_make(_TRUE_RAND, 0, 100, *a, **kw)
    if json_name is not None and hasattr(mod, "_load_cases"):
        json_path = os.path.join(os.path.dirname(path), json_name)

        def _patched(json_path=json_path):
            with open(json_path, "r") as f:
                return [json.loads(line) for line in f if line.strip()]

        mod._load_cases = _patched
    # a few upstream prepare_inputs default device to 'npu' when None; force CPU
    for fn_name in ("get_inputs", "get_init_inputs_per_case"):
        if hasattr(mod, fn_name):
            _orig_fn = getattr(mod, fn_name)

            def _cpu_fn(param, device=None, _orig_fn=_orig_fn):
                return _orig_fn(param, device=torch.device("cpu"))

            setattr(mod, fn_name, _cpu_fn)
    return mod


def norm(a):
    if isinstance(a, torch.Tensor):
        return a.detach().cpu()
    if isinstance(a, (list, tuple)) and a and all(isinstance(x, torch.Tensor) for x in a):
        return [x.detach().cpu() for x in a]
    if hasattr(a, "item") and not isinstance(a, (bool, int, float, str, bytes)):
        try:
            return a.item()  # numpy scalar -> python scalar
        except Exception:
            pass
    return a


def stats_of(values):
    """Aggregate stats for one arg position from a list of tensors."""
    tensors = [v for v in values if isinstance(v, torch.Tensor) and not v.is_sparse]
    if not tensors:
        return None
    dt = tensors[0].dtype
    if dt == torch.bool:
        allv = torch.cat([t.reshape(-1) for t in tensors])
        return {"kind": "bool", "true_frac": allv.to(torch.float32).mean().item(), "nf": 0.0}
    if dt in (torch.complex64, torch.complex128):
        tensors = [t.real for t in tensors] + [t.imag for t in tensors]
    if tensors[0].is_floating_point() or tensors[0].dtype in (torch.float16, torch.bfloat16):
        v = torch.cat([t.reshape(-1).to(torch.float64) for t in tensors])
        if v.numel() == 0:
            return {"kind": "float", "min": 0, "max": 0, "mean": 0, "std": 0, "nf": 0.0}
        nf = (~torch.isfinite(v)).to(torch.float64).mean().item()
        v = v[torch.isfinite(v)]
        if v.numel() == 0:
            return {"kind": "float", "min": 0, "max": 0, "mean": 0, "std": 0, "nf": nf}
        return {"kind": "float", "min": v.min().item(), "max": v.max().item(),
                "mean": v.mean().item(), "std": v.std().item(), "nf": nf}
    # int
    v = torch.cat([t.reshape(-1).to(torch.int64) for t in tensors])
    uniq = torch.unique(v).numel() / max(v.numel(), 1)
    if v.numel() == 0:
        return {"kind": "int", "min": 0, "max": 0, "uniq": 1.0, "nf": 0.0}
    return {"kind": "int", "min": int(v.min()), "max": int(v.max()), "uniq": uniq, "nf": 0.0}


def cmp_stats(new, old):
    """Return list of mismatch strings (empty = ok)."""
    msgs = []
    if new is None or old is None:
        return msgs
    if new["kind"] != old["kind"]:
        return ["kind %s vs %s" % (new["kind"], old["kind"])]
    if abs(new.get("nf", 0.0) - old.get("nf", 0.0)) > 0.02:
        msgs.append("nonfinite-frac %.3f vs %.3f" % (new.get("nf", 0), old.get("nf", 0)))
    if new.get("nf", 0) > 0.005 or old.get("nf", 0) > 0.005:
        return msgs  # non-finite present: numeric moments not comparable
    if new["kind"] == "bool":
        if abs(new["true_frac"] - old["true_frac"]) > 0.05:
            msgs.append("true_frac %.3f vs %.3f" % (new["true_frac"], old["true_frac"]))
    elif new["kind"] == "int":
        if not (old["min"] <= new["min"] and new["max"] <= old["max"]):
            # new bounds outside original observed bounds
            msgs.append("int range [%d,%d] vs orig [%d,%d]" % (new["min"], new["max"], old["min"], old["max"]))
        if old["uniq"] > 0.98 and new["uniq"] < 0.9:
            msgs.append("uniq-ratio %.3f vs %.3f" % (new["uniq"], old["uniq"]))
    else:
        for k in ("mean", "std"):
            a, b = new[k], old[k]
            if abs(a - b) > STAT_ATOL + STAT_RTOL * abs(b):
                msgs.append("%s %.4g vs orig %.4g" % (k, a, b))
        # support containment with slack
        if new["min"] < old["min"] - max(1.0, 0.1 * abs(old["min"])) or \
           new["max"] > old["max"] + max(1.0, 0.1 * abs(old["max"])):
            msgs.append("float range [%.4g,%.4g] vs orig [%.4g,%.4g]"
                        % (new["min"], new["max"], old["min"], old["max"]))
    return msgs


def structural_diff(new_groups, old_groups, random_attrs=None, shape_unstable=None):
    msgs = []
    if len(new_groups) != len(old_groups):
        return ["case count %d vs %d" % (len(new_groups), len(old_groups))]
    for c, (ng, og) in enumerate(zip(new_groups, old_groups)):
        if len(ng) != len(og):
            msgs.append("case %d: arg count %d vs %d" % (c, len(ng), len(og)))
            continue
        for j, (na, oa) in enumerate(zip(ng, og)):
            if random_attrs and (c, j) in random_attrs:
                lo, hi = random_attrs[(c, j)]
                if isinstance(na, (int, float)) and not (lo - 1e-9 <= na <= hi + 1e-9):
                    msgs.append("case %d arg %d: random attr %r outside [%r, %r]" % (c, j, na, lo, hi))
                continue
            if shape_unstable and (c, j) in shape_unstable and \
                    isinstance(na, torch.Tensor) and isinstance(oa, torch.Tensor):
                if na.dtype != oa.dtype:
                    msgs.append("case %d arg %d: dtype %s vs %s" % (c, j, na.dtype, oa.dtype))
                continue
            msgs.extend(arg_diff("case %d arg %d" % (c, j), na, oa))
        if len(msgs) > 20:
            msgs.append("... truncated")
            break
    return msgs


def arg_diff(where, na, oa):
    if isinstance(oa, torch.Tensor) or isinstance(na, torch.Tensor):
        if not (isinstance(oa, torch.Tensor) and isinstance(na, torch.Tensor)):
            return ["%s: type %s vs %s" % (where, type(na).__name__, type(oa).__name__)]
        if na.is_sparse != oa.is_sparse:
            return ["%s: sparse %s vs %s" % (where, na.is_sparse, oa.is_sparse)]
        if na.dtype != oa.dtype:
            return ["%s: dtype %s vs %s" % (where, na.dtype, oa.dtype)]
        if list(na.shape) != list(oa.shape):
            return ["%s: shape %s vs %s" % (where, list(na.shape), list(oa.shape))]
        return []
    if isinstance(oa, list) or isinstance(na, list):
        if not (isinstance(oa, list) and isinstance(na, list)) or len(na) != len(oa):
            return ["%s: list mismatch %s vs %s" % (where, type(na).__name__, type(oa).__name__)]
        if oa and all(isinstance(x, torch.Tensor) for x in oa):
            out = []
            for k, (x, y) in enumerate(zip(na, oa)):
                out.extend(arg_diff("%s[%d]" % (where, k), x, y))
            return out
        return [] if na == oa else ["%s: list value %r vs %r" % (where, na, oa)]
    if oa is None or na is None:
        return [] if (oa is None and na is None) else \
            ["%s: %r vs %r" % (where, na, oa)]
    # scalar attr
    if type(na) is not type(oa) and not (isinstance(na, (int, float)) and isinstance(oa, (int, float))):
        return ["%s: attr type %s vs %s" % (where, type(na).__name__, type(oa).__name__)]
    if na != oa:
        return ["%s: attr value %r vs %r" % (where, na, oa)]
    return []


def run_forward(Model, init, args):
    """no_grad first; on failure retry with float inputs requiring grad."""
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


def finite_pattern(out):
    pats = []
    outs = out if isinstance(out, (tuple, list)) else [out]
    for o in outs:
        if isinstance(o, torch.Tensor) and (o.is_floating_point() or o.is_complex()):
            pats.append(bool(torch.isfinite(o).all()))
    return pats


def verify_op(level, old_id, op_name, new_base):
    res = {"op": new_base, "status": "PASS", "struct": [], "review": [], "error": None}
    src_py = os.path.join(SRC_DIR, level, "%d_%s.py" % (old_id, op_name))
    old = load_module(src_py, "old_" + new_base, json_name="%d_%s.json" % (old_id, op_name))
    new = load_module(os.path.join(OUT_DIR, new_base + ".py"), "new_" + new_base)

    old_groups = [[norm(a) for a in g] for g in old.get_input_groups()]
    old_inits = [[norm(a) for a in g] for g in old.get_init_inputs()]
    new_groups = [[norm(a) for a in g] for g in new.get_input_groups()]
    new_inits = [[norm(a) for a in g] for g in new.get_init_inputs()]

    # intentionally dropped cases (e.g. NPU-unsupported dtypes): exclude the
    # same case indices from the original side before all comparisons
    try:
        from convert import OP_OVERRIDES
        drop = set(OP_OVERRIDES.get(op_name, {}).get("drop_cases", []))
    except Exception:
        drop = set()
    if drop:
        old_groups = [g for c, g in enumerate(old_groups) if c not in drop]
        old_inits = [g for c, g in enumerate(old_inits) if c not in drop]

    # per-case json metadata: random attrs (exempt from exact compare) and
    # value-stored tensors (skip distribution check -- they ARE original values)
    random_attrs = {}
    data_args = set()
    shape_derived = set()
    try:
        with open(os.path.join(OUT_DIR, new_base + ".json")) as f:
            json_cases = [json.loads(l) for l in f if l.strip()]
        for c, case in enumerate(json_cases):
            for j, e in enumerate(case.get("inputs", [])):
                if e.get("random") and "range" in e:
                    random_attrs[(c, j)] = (e["range"][0], e["range"][1])
                if e.get("shape_derived"):
                    shape_derived.add((c, j))
        if json_cases:
            n_args_j = len(json_cases[0].get("inputs", []))
            for j in range(n_args_j):
                if all(j < len(cs.get("inputs", [])) and "data" in cs["inputs"][j]
                       for cs in json_cases):
                    data_args.add(j)
    except Exception:
        pass

    # 1. structural; the original generator may produce random shapes (derived
    # from random values) -- detect by sampling twice and exempt those positions
    old_groups_b = [[norm(a) for a in g] for g in old.get_input_groups()]
    if drop:
        old_groups_b = [g for c, g in enumerate(old_groups_b) if c not in drop]
    shape_unstable = set(shape_derived)
    for c, (ga, gb) in enumerate(zip(old_groups, old_groups_b)):
        for j, (aa, ab) in enumerate(zip(ga, gb)):
            if isinstance(aa, torch.Tensor) and isinstance(ab, torch.Tensor) \
                    and list(aa.shape) != list(ab.shape):
                shape_unstable.add((c, j))
    res["struct"] = structural_diff(new_groups, old_groups, random_attrs, shape_unstable)
    if not res["struct"] and any(len(g) for g in old_inits):
        res["struct"] = structural_diff(new_inits, old_inits)

    # 2. forward on new inputs + finiteness comparison
    fwd_fail = []
    for c, args in enumerate(new_groups):
        init = new_inits[c] if new_inits else []
        out_new, err = run_forward(new.Model, init, args)
        if err is not None:
            fwd_fail.append("case %d: %r" % (c, err))
            if len(fwd_fail) >= 3:
                break
            continue
        o_init = old_inits[c] if old_inits else []
        out_old, err2 = run_forward(old.Model, o_init, old_groups[c])
        if err2 is None and finite_pattern(out_new) != finite_pattern(out_old):
            fwd_fail.append("case %d: finite pattern %s vs orig %s"
                            % (c, finite_pattern(out_new), finite_pattern(out_old)))
        if len(fwd_fail) >= 3:
            break

    # 3. distribution
    dist_msgs = []
    if not res["struct"]:
        new_samples = [new.get_input_groups() for _ in range(SAMPLES - 1)] + [new_groups]
        old_samples = [old.get_input_groups() for _ in range(SAMPLES - 1)] + [old_groups]
        if drop:
            old_samples = [[g for c, g in enumerate(s) if c not in drop] for s in old_samples]
        n_args = len(new_groups[0]) if new_groups else 0
        for j in range(n_args):
            if j in data_args:
                continue  # value-stored: identical to original by construction
            nv = [norm(s[c][j]) for s in new_samples for c in range(len(new_groups))]
            ov = [norm(s[c][j]) for s in old_samples for c in range(len(old_groups))]
            m = cmp_stats(stats_of(nv), stats_of(ov))
            if m:
                dist_msgs.append("arg %d (%s): %s" % (j, _arg_name(new, j), "; ".join(m)))
        del new_samples, old_samples

    if res["struct"] or fwd_fail:
        res["status"] = "FAIL"
        res["error"] = (res["struct"] + fwd_fail)[:6]
    elif dist_msgs:
        res["status"] = "REVIEW"
        res["review"] = dist_msgs
    del old, new, old_groups, new_groups, old_inits, new_inits
    gc.collect()
    return res


def _arg_name(mod, j):
    import inspect
    names = [p.name for p in inspect.signature(mod.Model.forward).parameters.values()
             if p.name != "self"]
    return names[j] if j < len(names) else "arg%d" % j


def main():
    manifest = json.load(open(os.path.join(OUT_DIR, "..", "report", "_manifest.json")))
    only = sys.argv[1:] if len(sys.argv) > 1 else None
    results = []
    for m in manifest:
        if m["status"] != "ok":
            results.append({"op": "%s/%d_%s" % (m["level"], m["old_id"], m["op"]),
                            "status": "FAIL", "error": ["convert: " + m.get("error", "?")],
                            "struct": [], "review": []})
            continue
        if only and m["new"] not in only:
            continue
        try:
            r = verify_op(m["level"], m["old_id"], m["op"], m["new"])
        except Exception as e:
            r = {"op": m["new"], "status": "FAIL", "struct": [], "review": [],
                 "error": ["verify crash: " + repr(e), traceback.format_exc(limit=3)]}
        results.append(r)
        print("[%s] %s" % (r["status"], r["op"]), flush=True)

    n_pass = sum(1 for r in results if r["status"] == "PASS")
    n_rev = sum(1 for r in results if r["status"] == "REVIEW")
    n_fail = sum(1 for r in results if r["status"] == "FAIL")
    lines = ["# CANN_Ops 入库验证报告", "",
             "| 状态 | 数量 |", "|---|---|",
             "| PASS | %d |" % n_pass, "| REVIEW | %d |" % n_rev, "| FAIL | %d |" % n_fail, ""]
    for st in ("FAIL", "REVIEW"):
        rows = [r for r in results if r["status"] == st]
        if not rows:
            continue
        lines.append("## %s (%d)" % (st, len(rows)))
        lines.append("")
        lines.append("| 算子 | 详情 |")
        lines.append("|---|---|")
        for r in rows:
            detail = "<br>".join(str(x) for x in (r["error"] or r["review"])[:6])
            lines.append("| %s | %s |" % (r["op"], detail.replace("|", "\\|")))
        lines.append("")
    with open(REPORT, "w") as f:
        f.write("\n".join(lines))
    print("PASS %d / REVIEW %d / FAIL %d -> %s" % (n_pass, n_rev, n_fail, REPORT))
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
