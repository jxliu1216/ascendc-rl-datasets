#!/usr/bin/env python3
"""Convert cann_ops_tmp/{level1,2,3}/{oldid}_{Op}.py + .json into CANN_Ops/.

Output per operator (flat dir, ids renumbered 0-based within each level,
ordered by original numeric id):
    cannops_level{N}_{newid}_{OpName}.py    NPUKernelBench-style reference impl
    cannops_level{N}_{newid}_{OpName}.json  typed {"inputs": [...]} JSON Lines

Strategy: execute the original module's get_input_groups()/get_init_inputs(),
sample each case several times, classify every argument's generation recipe
(randn / rand / bounded randint / randperm / arange / bool / complex / attr...),
serialize the typed json from the concrete values, and emit per-op
construction code in the NPUKernelBench idiom (hardcoded same-name json,
positional construction).
cann_ops_tmp/ is treated read-only.
"""

import gc
import importlib.util
import inspect
import json
import math
import os
import re
import sys

import torch

SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "cann_ops_tmp")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "CANN_Ops")
SAMPLES = 3  # generator samples per op for recipe classification

FLOAT_DTYPES = {torch.float16, torch.float32, torch.float64, torch.bfloat16}
INT_DTYPES = {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8}
COMPLEX_DTYPES = {torch.complex64, torch.complex128}

DTYPE_NAME = {
    torch.float16: "float16", torch.float32: "float32", torch.float64: "float64",
    torch.bfloat16: "bfloat16", torch.int8: "int8", torch.int16: "int16",
    torch.int32: "int32", torch.int64: "int64", torch.uint8: "uint8",
    torch.bool: "bool", torch.complex64: "complex64", torch.complex128: "complex128",
}


# ---------------------------------------------------------------- sampling

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


def _canon_randn(*a, **kw):
    return _safe_make(_TRUE_RANDN, -100, 100, *a, **kw)


def _canon_rand(*a, **kw):
    return _safe_make(_TRUE_RAND, 0, 100, *a, **kw)


def load_module(path, tag, json_name=None):
    name = "_cannops_src_%s" % tag
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    # each source file re-wraps torch.randn/rand on import, capturing the
    # previous wrapper -> O(n) call chain after n imports. Reset to a single
    # canonical wrapper before/after each load so depth stays 1.
    torch.randn, torch.rand = _TRUE_RANDN, _TRUE_RAND
    spec.loader.exec_module(mod)
    torch.randn, torch.rand = _canon_randn, _canon_rand
    if json_name is not None and hasattr(mod, "_load_cases"):
        # source files hardcode "{OpName}.json" but the file on disk is
        # "{oldid}_{OpName}.json"; cann_ops_tmp is read-only, so patch the loader
        json_path = os.path.join(os.path.dirname(path), json_name)

        def _patched_load_cases(json_path=json_path):
            with open(json_path, "r") as f:
                return [json.loads(line) for line in f if line.strip()]

        mod._load_cases = _patched_load_cases
    # a few upstream prepare_inputs default device to 'npu' when None; force CPU
    for fn_name in ("get_inputs", "get_init_inputs_per_case"):
        if hasattr(mod, fn_name):
            _orig_fn = getattr(mod, fn_name)

            def _cpu_fn(param, device=None, _orig_fn=_orig_fn):
                return _orig_fn(param, device=torch.device("cpu"))

            setattr(mod, fn_name, _cpu_fn)
    return mod


def norm_arg(a):
    """Normalize one generated argument."""
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


def sample_op(mod):
    """Return (input_samples, init_groups): input_samples[s][case][arg]."""
    input_samples = []
    for _ in range(SAMPLES):
        groups = mod.get_input_groups()
        input_samples.append([[norm_arg(a) for a in case] for case in groups])
        del groups
    init_groups = [[norm_arg(a) for a in case] for case in mod.get_init_inputs()]
    gc.collect()
    return input_samples, init_groups


# ---------------------------------------------------------------- recipe classification

class Recipe:
    """Generation recipe for one argument position (aggregated over cases/samples)."""

    def __init__(self, name):
        self.name = name
        self.kind = None          # randn|rand|rand_range|randint|randperm|arange|
                                  # bool_rand|bool_full|complex|attr|none|tensor_list|mixed
        self.dtype = None         # tensor dtype (torch.dtype) or attr dtype str
        self.bool_value = None    # for bool_full
        self.sub = None           # complex: (re_recipe, im_recipe)
        self.elem = None          # tensor_list: list of element Recipes
        self.mixed = None         # mixed: {"tensor": Recipe, "none": Recipe}


def _float_stats(tensors):
    vals = torch.cat([t.reshape(-1).to(torch.float64) for t in tensors])
    if vals.numel() == 0:
        return dict(min=0.0, max=0.0, mean=0.0, std=1.0, integer_valued=False)
    return dict(
        min=vals.min().item(), max=vals.max().item(),
        mean=vals.mean().item(), std=vals.std().item(),
        integer_valued=bool(torch.equal(vals, vals.round())),
    )


def _classify_float(name, tensors, per_case_tensors=None):
    """Classify a float-arg position. When per_case_tensors is given, the
    uniform-vs-normal test is done per case (median) so that cases with
    different scales don't pollute each other's pooled statistics."""
    st = _float_stats(tensors)
    r = Recipe(name)
    if st["min"] == st["max"]:
        r.kind = "const"
        r.const = st["min"]
        return r

    def _ratio(ts):
        s = _float_stats(ts)
        span = max(s["max"] - s["min"], 1e-9)
        return s["std"] / span, s

    if per_case_tensors:
        ratios = sorted(_ratio(ts)[0] for ts in per_case_tensors)
        ratio = ratios[len(ratios) // 2]
        per_case = [_ratio(ts)[1] for ts in per_case_tensors]
        # per-case tests, then majority / median
        frac_rand01 = sum(1 for s in per_case
                          if s["min"] >= -1e-5 and s["max"] <= 1.0 + 1e-5
                          and not s["integer_valued"]) / len(per_case)
        frac_pos = sum(1 for s in per_case if s["min"] >= -1e-5) / len(per_case)
        frac_std_norm = sum(1 for s in per_case
                            if abs(s["mean"]) < 0.15 and abs(s["std"] - 1.0) < 0.15
                            ) / len(per_case)
        if frac_rand01 > 0.5:
            r.kind = "rand"
        elif frac_pos > 0.5:
            r.kind = "rand_range"
        elif ratio > 0.22:
            # std/(max-min) ~ 1/sqrt(12)=0.289 uniform, ~0.167 normal
            r.kind = "rand_range"
        elif frac_std_norm > 0.5:
            r.kind = "randn"
        else:
            r.kind = "randn_scaled"
            r.mean, r.std = st["mean"], max(st["std"], 1e-3)
        return r

    if st["min"] >= -1e-5 and st["max"] <= 1.0 + 1e-5 and not st["integer_valued"]:
        r.kind = "rand"
    elif st["min"] >= -1e-5:
        r.kind = "rand_range"
    else:
        ratio = st["std"] / max(st["max"] - st["min"], 1e-9)
        if ratio > 0.22:
            r.kind = "rand_range"
        elif abs(st["mean"]) < 0.15 and abs(st["std"] - 1.0) < 0.15:
            r.kind = "randn"
        else:
            r.kind = "randn_scaled"
            r.mean, r.std = st["mean"], max(st["std"], 1e-3)
    return r


def _classify_int(name, per_case_tensors):
    """per_case_tensors: list over cases of [sample tensors]."""
    r = Recipe(name)
    all_flat = [t.reshape(-1).to(torch.int64) for samples in per_case_tensors for t in samples]
    if all_flat:
        lo_g = int(min(t.min() for t in all_flat if t.numel())) if any(t.numel() for t in all_flat) else 0
        hi_g = int(max(t.max() for t in all_flat if t.numel())) if any(t.numel() for t in all_flat) else 0
        if lo_g == hi_g:
            r.kind = "const"
            r.const = lo_g
            return r
    # permutation / arange detection: consistent across every case & sample
    is_arange, is_perm = True, True
    for samples in per_case_tensors:
        for t in samples:
            flat = t.reshape(-1).to(torch.int64)
            n = flat.numel()
            if t.dim() != 1 or n == 0:
                is_arange = is_perm = False
                continue
            srt = torch.sort(flat).values
            lo = int(flat.min())
            if not torch.equal(flat, torch.arange(lo, lo + n, dtype=torch.int64)):
                is_arange = False
            if not torch.equal(srt, torch.arange(lo, lo + n, dtype=torch.int64)):
                is_perm = False
    if is_arange:
        r.kind = "arange"
        return r
    if is_perm:
        r.kind = "randperm"
        return r
    r.kind = "randint"
    return r


def classify_position(name, samples_by_case):
    """samples_by_case: list over cases of list over samples of one normalized arg."""
    flat = [a for samples in samples_by_case for a in samples]
    kinds = set()
    for a in flat:
        kinds.add(_kind_of(a))
    if len(kinds) > 1:
        r = Recipe(name)
        r.kind = "mixed"
        r.mixed = {}
        for k in kinds:
            vals = [a for a in flat if _kind_of(a) == k]
            r.mixed[k] = _recipe_from_values(name, k, vals)
        return r
    return _recipe_from_values(name, kinds.pop(), flat, samples_by_case)


def _kind_of(a):
    if isinstance(a, torch.Tensor):
        return "sparse" if a.is_sparse else "tensor"
    if isinstance(a, list):
        if a and all(isinstance(x, torch.Tensor) for x in a):
            return "tensor_list"
        return "attr"
    if a is None:
        return "none"
    return "attr"


def _dt_class(dt):
    if dt in FLOAT_DTYPES:
        return "float"
    if dt in INT_DTYPES:
        return "int"
    if dt == torch.bool:
        return "bool"
    if dt in COMPLEX_DTYPES:
        return "complex"
    return "other"


def _recipe_from_values(name, kind, values, samples_by_case=None):
    r = Recipe(name)
    if kind == "none":
        r.kind = "none"
        return r
    if kind == "sparse":
        r.kind = "sparse"
        r.dtype = values[0].dtype
        return r
    if kind == "attr":
        r.kind = "attr"
        r.dtype = _attr_dtype(values[0])
        # distinguish a real torch.dtype attr from a python bool named "bool" etc.
        r.is_torch_dtype = isinstance(values[0], torch.dtype)
        return r
    if kind == "tensor_list":
        r.kind = "tensor_list"
        # pool ALL elements across cases/positions: element dtype class may
        # vary per case (e.g. float16 vs int32) regardless of list length
        el_vals = [t for v in values for t in v]
        r.elem = [_recipe_from_values(name + "_el", "tensor", el_vals)]
        return r
    # tensor
    classes = {_dt_class(v.dtype) for v in values}
    if len(classes) > 1:
        # dtype varies per case across type classes (e.g. float32 and int32):
        # dispatch on the json entry's dtype at runtime
        r.kind = "poly"
        r.subs = {}
        for cls in classes:
            vals_c = [v for v in values if _dt_class(v.dtype) == cls]
            sbc_c = None
            if samples_by_case is not None:
                sbc_c = [[t for t in samples if isinstance(t, torch.Tensor)
                          and _dt_class(t.dtype) == cls] for samples in samples_by_case]
                sbc_c = [ts for ts in sbc_c if ts] or None
            r.subs[cls] = _recipe_from_values(name, "tensor", vals_c, sbc_c)
        return r
    dt = values[0].dtype
    r.dtype = dt
    if dt in FLOAT_DTYPES:
        per_case = None
        if samples_by_case is not None:
            per_case = [[t for t in samples if isinstance(t, torch.Tensor)
                         and t.dtype in FLOAT_DTYPES] for samples in samples_by_case]
            per_case = [ts for ts in per_case if ts]
            if len(per_case) < 2:
                per_case = None
        return _classify_float(name, values, per_case)
    if dt in INT_DTYPES:
        if samples_by_case is None:
            samples_by_case = [[v] for v in values]
        return _classify_int(name, samples_by_case)
    if dt == torch.bool:
        allv = torch.cat([v.reshape(-1) for v in values])
        if bool(allv.all()) or bool((~allv).all()):
            r.kind = "bool_full"
            r.bool_value = bool(allv[0])
        else:
            r.kind = "bool_rand"
            r.true_frac = allv.to(torch.float32).mean().item()
        return r
    if dt in COMPLEX_DTYPES:
        r.kind = "complex"
        r.sub = (_classify_float(name + ".real", [v.real for v in values]),
                 _classify_float(name + ".imag", [v.imag for v in values]))
        return r
    raise ValueError("unsupported dtype %s for %s" % (dt, name))


def _attr_dtype(v):
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, int):
        return "int"
    if isinstance(v, float):
        return "float"
    if isinstance(v, str):
        return "str"
    if isinstance(v, tuple):
        return "tuple"
    if isinstance(v, list):
        return "list"
    if isinstance(v, torch.dtype):
        return DTYPE_NAME.get(v, str(v).replace("torch.", ""))
    return type(v).__name__


# ---------------------------------------------------------------- json serialization

def _json_scalar(v):
    if isinstance(v, torch.dtype):
        return DTYPE_NAME.get(v, str(v).replace("torch.", ""))
    if isinstance(v, (tuple, list)):
        return [_json_scalar(x) for x in v]
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return str(v)
    if hasattr(v, "item"):
        return _json_scalar(v.item())
    return v


def _int_range(values):
    lo = min(int(v.min()) for v in values if v.numel()) if any(v.numel() for v in values) else 0
    hi = max(int(v.max()) for v in values if v.numel()) if any(v.numel() for v in values) else 0
    return [lo, hi]


def _tensor_value(v):
    """Flatten a small tensor to a JSON-safe list of scalars."""
    return v.detach().cpu().reshape(-1).tolist()


def _nonfinite_kind(values):
    """Detect non-finite values in sampled tensors; return 'nan'/'inf'/None."""
    for v in values:
        ts = v if isinstance(v, list) else [v]
        for t in ts:
            if isinstance(t, torch.Tensor) and not t.is_sparse and \
                    (t.is_floating_point() or t.is_complex()):
                if bool(torch.isnan(t).any()):
                    return "nan"
                if bool(torch.isinf(t).any()):
                    return "inf"
    return None


VALUE_STORE_MAX = 64  # tensors with numel <= this are stored by exact value ("data" key)
FORCE_VALUE_MAX = 300000  # cap for forced value storage (shape drivers, op overrides)

# per-op conversion overrides
OP_OVERRIDES = {
    "RoiAlignRotated": {"force_value_args": ["rois"]},
    "RoiAlignRotatedGrad": {"force_value_args": ["rois"]},
    # CPU aten._ctc_loss lacks bf16/fp16 kernels (original ran on NPU);
    # upcast to fp32 for the aten call, post-processing casts back
    "CTCLossV3": {"model_patch": [
        ("._ctc_loss(log_probs, targets,", "._ctc_loss(log_probs.float(), targets,")]},
    "CTCLossV3Grad": {"model_patch": [
        ("_ctc_loss_backward(grad, log_probs,", "_ctc_loss_backward(grad.float(), log_probs.float(),"),
        ("neg_log_likelihood, log_aplha, blank", "neg_log_likelihood.float(), log_aplha.float(), blank"),
        ("return res.cpu()", "return res.to(grad.dtype)")]},
    # sparse COO input is rebuilt from the indices/values args
    "CoalesceSparse": {"sparse_from": {"indices": "indices", "values": "values"}},
    # original replaces exact zeros with 1 (reciprocal domain)
    "ForeachReciprocal": {"nonzero_fix_args": ["inputs"]},
}


def serialize_entry(recipe, case_values, requires_grad):
    """Build one json entry. case_values: concrete values of this arg for this case."""
    v0 = case_values[0]
    if recipe.kind == "mixed":
        k = _kind_of(v0)
        sub = recipe.mixed[k]
        vals = case_values
        return serialize_entry(sub, vals, requires_grad) if k != "none" else {
            "name": recipe.name, "type": "attr", "required": False,
            "dtype": "none", "value": None}
    if recipe.kind == "none":
        return {"name": recipe.name, "type": "attr", "required": False,
                "dtype": "none", "value": None}
    if recipe.kind == "sparse":
        return {"name": recipe.name, "type": "sparse_tensor", "required": True,
                "dtype": DTYPE_NAME[v0.dtype], "shape": list(v0.shape)}
    if recipe.kind == "attr":
        entry = {"name": recipe.name, "type": "attr",
                 "required": True, "dtype": recipe.dtype, "value": _json_scalar(v0)}
        if getattr(recipe, "random", False):
            entry["random"] = True
            entry["range"] = [_json_scalar(x) for x in recipe.attr_range]
        return entry
    if recipe.kind == "tensor_list":
        shapes = [list(t.shape) for t in v0]
        entry = {"name": recipe.name, "type": "tensor_list", "required": True,
                 "dtype": DTYPE_NAME[v0[0].dtype], "shapes": shapes}
        el = recipe.elem[0] if recipe.elem else None
        if el is not None and el.kind == "poly":
            cls = _dt_class(v0[0].dtype)
            if cls == "int":
                entry["range"] = _int_range(list(v0))
            elif cls == "float":
                sub = el.subs.get("float")
                if sub is not None and sub.kind == "rand_range":
                    st = _float_stats(list(v0))
                    entry["range"] = [max(st["min"], 0.0), max(st["max"], 1.0)]
                elif sub is not None and sub.kind == "randn_scaled":
                    entry["mean"], entry["std"] = sub.mean, sub.std
            return entry
        if el is not None and el.kind in ("randint", "randperm", "arange"):
            entry["range"] = _int_range([t for t in v0])
        if el is not None and el.kind == "rand_range":
            st = _float_stats(list(v0))
            entry["range"] = [max(st["min"], 0.0), max(st["max"], 1.0)]
        if el is not None and el.kind == "randn_scaled":
            entry["mean"], entry["std"] = el.mean, el.std
        nf = _nonfinite_kind(case_values)
        if nf:
            entry["inject"] = nf
            recipe.inject_seen = True
        return entry
    if recipe.kind == "poly":
        cls = _dt_class(v0.dtype)
        entry = {"name": recipe.name, "type": "tensor", "required": True,
                 "dtype": DTYPE_NAME[v0.dtype], "shape": list(v0.shape)}
        if v0.numel() <= VALUE_STORE_MAX and cls != "complex":
            entry["data"] = _tensor_value(v0)
            if requires_grad:
                entry["requires_grad"] = True
            return entry
        cls_vals = [v for v in case_values if isinstance(v, torch.Tensor)
                    and _dt_class(v.dtype) == cls]
        sub = recipe.subs[cls]
        if cls == "int":
            if sub.kind == "const":
                entry["fill"] = _json_scalar(sub.const)
            else:
                entry["range"] = _int_range(cls_vals)
        elif cls == "bool" and sub.kind == "bool_full":
            entry["value"] = sub.bool_value
        elif cls == "bool" and sub.kind == "bool_rand":
            entry["true_frac"] = sub.true_frac
        elif cls == "float":
            if sub.kind == "const":
                entry["fill"] = _json_scalar(sub.const)
            elif sub.kind == "rand_range":
                st = _float_stats(cls_vals)
                entry["range"] = [st["min"], st["max"]]
            elif sub.kind == "randn_scaled":
                entry["mean"], entry["std"] = sub.mean, sub.std
        nf = _nonfinite_kind(case_values)
        if nf:
            entry["inject"] = nf
            recipe.inject_seen = True
        if requires_grad:
            entry["requires_grad"] = True
        return entry
    # tensor
    entry = {"name": recipe.name, "type": "tensor", "required": True,
             "dtype": DTYPE_NAME[v0.dtype], "shape": list(v0.shape)}
    if v0.numel() <= VALUE_STORE_MAX and v0.dtype not in COMPLEX_DTYPES:
        entry["data"] = _tensor_value(v0)
        if requires_grad:
            entry["requires_grad"] = True
        return entry
    if recipe.kind in ("randint", "randperm", "arange"):
        entry["range"] = _int_range(case_values)
    if recipe.kind == "const":
        entry["fill"] = _json_scalar(recipe.const)
    if recipe.kind == "rand_range":
        st = _float_stats(case_values)
        entry["range"] = [st["min"], st["max"]]
    if recipe.kind == "randn_scaled":
        entry["mean"], entry["std"] = recipe.mean, recipe.std
    if recipe.kind == "bool_full":
        entry["value"] = recipe.bool_value
    if recipe.kind == "bool_rand":
        entry["true_frac"] = recipe.true_frac
    nf = _nonfinite_kind(case_values)
    if nf:
        entry["inject"] = nf
        recipe.inject_seen = True
    if requires_grad:
        entry["requires_grad"] = True
    return entry


# ---------------------------------------------------------------- py code generation

HEADER = """import torch
import torch.nn as nn
import json
import os

DTYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
    "bfloat16": torch.bfloat16,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "uint8": torch.uint8,
    "bool": torch.bool,
    "complex64": torch.complex64,
}

"""


def gen_tensor_expr(var, info, recipe, indent=""):
    """Emit lines constructing `var` from json dict `info` per recipe."""
    L = []
    dt = "DTYPE_MAP[%s[\"dtype\"]]" % info
    shape = "%s[\"shape\"]" % info
    kind = recipe.kind
    if kind == "randn":
        L.append("%s = torch.randn(%s, dtype=%s)" % (var, shape, dt))
    elif kind == "const":
        L.append("%s = torch.full(%s, %s[\"fill\"], dtype=%s)" % (var, shape, info, dt))
    elif kind == "rand":
        L.append("%s = torch.rand(%s, dtype=%s)" % (var, shape, dt))
    elif kind == "randn_scaled":
        L.append("%s = torch.randn(%s, dtype=%s) * %s[\"std\"] + %s[\"mean\"]" % (var, shape, dt, info, info))
    elif kind == "rand_range":
        L.append("%s = torch.rand(%s, dtype=%s) * (%s[\"range\"][1] - %s[\"range\"][0]) + %s[\"range\"][0]"
                 % (var, shape, dt, info, info, info))
    elif kind == "randint":
        L.append("%s = torch.randint(%s[\"range\"][0], %s[\"range\"][1] + 1, tuple(%s), dtype=%s)"
                 % (var, info, info, shape, dt))
    elif kind == "randperm":
        L.append("%s = torch.randperm(%s[0], dtype=%s) + %s[\"range\"][0]" % (var, shape, dt, info))
    elif kind == "arange":
        L.append("%s = torch.arange(%s[\"range\"][0], %s[\"range\"][0] + %s[0], dtype=%s).reshape(%s)"
                 % (var, info, info, shape, dt, shape))
    elif kind == "bool_rand":
        L.append("%s = torch.rand(%s) < %s.get(\"true_frac\", 0.5)" % (var, shape, info))
    elif kind == "bool_full":
        L.append("%s = torch.full(%s, %s[\"value\"], dtype=torch.bool)" % (var, shape, info))
    elif kind == "complex":
        re, im = recipe.sub
        part_dt = {torch.complex64: "torch.float32",
                   torch.complex128: "torch.float64"}.get(recipe.dtype, "torch.float32")
        for suffix, sub in (("_re", re), ("_im", im)):
            if sub.kind == "rand":
                L.append("%s%s = torch.rand(%s, dtype=%s)" % (indent, var + suffix, shape, part_dt))
            else:
                L.append("%s%s = torch.randn(%s, dtype=%s)" % (indent, var + suffix, shape, part_dt))
        L.append("%s%s = torch.complex(%s_re, %s_im).to(%s)" % (indent, var, var, var, dt))
    else:
        raise ValueError("no tensor expr for kind %s" % kind)
    return L


def gen_arg_lines(var, info, recipe, indent="        "):
    """Full per-arg code (handles mixed/none/attr/tensor/tensor_list)."""
    L = []
    if recipe.kind == "mixed":
        subs = recipe.mixed
        L.append("%sif %s[\"type\"] == \"attr\":" % (indent, info))
        L.append("%s    if %s.get(\"dtype\") == \"none\":" % (indent, info))
        L.append("%s        %s = None" % (indent, var))
        attr_sub = subs.get("attr")
        L.append("%s    else:" % indent)
        if attr_sub is not None and attr_sub.kind == "attr":
            for l in gen_arg_lines(var, info, attr_sub, ""):
                L.append("%s        %s" % (indent, l))
        else:
            L.append("%s        %s = %s[\"value\"]" % (indent, var, info))
        for k in ("tensor", "tensor_list"):
            if k in subs:
                L.append("%selse:" % indent)
                L.extend(gen_arg_lines(var, info, subs[k], indent + "    "))
        return L
    if recipe.kind == "none":
        L.append("%s%s = None" % (indent, var))
        return L
    if recipe.kind == "sparse":
        sf = getattr(recipe, "sparse_from", None) or {}
        L.append("%s%s = torch.sparse_coo_tensor(%s, %s, %s[\"shape\"])"
                 % (indent, var, sf.get("indices", "indices"), sf.get("values", "values"), info))
        return L
    if recipe.kind == "attr":
        if getattr(recipe, "is_torch_dtype", False):
            # attr carrying a torch dtype (e.g. dtype="float16")
            L.append("%s%s = DTYPE_MAP[%s[\"value\"]]" % (indent, var, info))
        elif getattr(recipe, "random", False):
            if recipe.dtype == "int":
                L.append("%s%s = torch.randint(%s[\"range\"][0], %s[\"range\"][1] + 1, (1,)).item()"
                         % (indent, var, info, info))
            else:
                L.append("%s%s = torch.rand(1).item() * (%s[\"range\"][1] - %s[\"range\"][0]) + %s[\"range\"][0]"
                         % (indent, var, info, info, info))
        elif recipe.dtype == "tuple":
            L.append("%s%s = tuple(%s[\"value\"])" % (indent, var, info))
        else:
            L.append("%s%s = %s[\"value\"]" % (indent, var, info))
        return L
    if recipe.kind == "tensor_list":
        L.append("%s%s = []" % (indent, var))
        L.append("%sfor _shape in %s[\"shapes\"]:" % (indent, info))
        el = recipe.elem[0]
        el_info = ('{"dtype": %s["dtype"], "shape": _shape, '
                   '"range": %s.get("range", [0, 1]), '
                   '"mean": %s.get("mean", 0.0), "std": %s.get("std", 1.0), '
                   '"value": %s.get("value")}' % (info, info, info, info, info))
        if el.kind in ("poly", "mixed"):
            el_lines = gen_arg_lines("_t", el_info, el, indent="")
        else:
            el_lines = gen_tensor_expr("_t", el_info, el)
        L.extend(["%s    %s" % (indent, l) for l in el_lines])
        L.append("%s    %s.append(_t)" % (indent, var))
        if getattr(recipe, "nonzero_fix", False):
            L.append("%sfor _i in range(len(%s)):" % (indent, var))
            L.append("%s    %s[_i][%s[_i] == 0] = 1" % (indent, var, var))
        if getattr(recipe, "inject_seen", False):
            L.append("%sif %s.get(\"inject\"):" % (indent, info))
            L.append("%s    _f = %s[0].reshape(-1)" % (indent, var))
            L.append("%s    _f[0] = float(%s[\"inject\"])" % (indent, info))
            L.append("%s    %s[0] = _f.reshape(%s[0].shape)" % (indent, var, var))
        return L
    if recipe.kind == "poly":
        shape = "%s[\"shape\"]" % info
        L.append("%sif \"data\" in %s:" % (indent, info))
        L.append("%s    %s = torch.tensor(%s[\"data\"], dtype=DTYPE_MAP[%s[\"dtype\"]]).reshape(%s)"
                 % (indent, var, info, info, shape))
        L.append("%selse:" % indent)
        L.append("%s    _dt = DTYPE_MAP[%s[\"dtype\"]]" % (indent, info))
        sub_ind = indent + "    "
        int_sub = recipe.subs.get("int")
        if int_sub is not None and int_sub.kind == "const":
            L.append("%sif _dt in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):" % sub_ind)
            L.append("%s    %s = torch.full(%s, %s[\"fill\"], dtype=_dt)" % (sub_ind, var, shape, info))
        else:
            L.append("%sif _dt in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):" % sub_ind)
            L.append("%s    %s = torch.randint(%s[\"range\"][0], %s[\"range\"][1] + 1, tuple(%s), dtype=_dt)"
                     % (sub_ind, var, info, info, shape))
        L.append("%selif _dt == torch.bool:" % sub_ind)
        if recipe.subs.get("bool") and recipe.subs["bool"].kind == "bool_full":
            L.append("%s    %s = torch.full(%s, %s[\"value\"], dtype=torch.bool)" % (sub_ind, var, shape, info))
        else:
            L.append("%s    %s = torch.rand(%s) < %s.get(\"true_frac\", 0.5)" % (sub_ind, var, shape, info))
        L.append("%selse:" % sub_ind)
        # float-class expression using _dt (randn also works for complex dtypes)
        fr = recipe.subs.get("float")
        if fr is None:
            fr = Recipe(var)
            fr.kind = "randn"
        if fr.kind == "rand":
            L.append("%s    %s = torch.rand(%s, dtype=_dt)" % (sub_ind, var, shape))
        elif fr.kind == "const":
            L.append("%s    %s = torch.full(%s, %s[\"fill\"], dtype=_dt)" % (sub_ind, var, shape, info))
        elif fr.kind == "rand_range":
            L.append("%s    %s = torch.rand(%s, dtype=_dt) * (%s[\"range\"][1] - %s[\"range\"][0]) + %s[\"range\"][0]"
                     % (sub_ind, var, shape, info, info, info))
        elif fr.kind == "randn_scaled":
            L.append("%s    %s = torch.randn(%s, dtype=_dt) * %s[\"std\"] + %s[\"mean\"]"
                     % (sub_ind, var, shape, info, info))
        else:
            L.append("%s    %s = torch.randn(%s, dtype=_dt)" % (sub_ind, var, shape))
        if getattr(recipe, "inject_seen", False):
            L.append("%sif %s.get(\"inject\"):" % (indent, info))
            L.append("%s    _f = %s.reshape(-1)" % (indent, var))
            L.append("%s    _f[0] = float(%s[\"inject\"])" % (indent, info))
            L.append("%s    %s = _f.reshape(%s.shape)" % (indent, var, var))
        return L
    # plain tensor
    L.append("%sif \"data\" in %s:" % (indent, info))
    L.append("%s    %s = torch.tensor(%s[\"data\"], dtype=DTYPE_MAP[%s[\"dtype\"]]).reshape(%s)"
             % (indent, var, info, info, "%s[\"shape\"]" % info))
    L.append("%selse:" % indent)
    L.extend([indent + "    " + l for l in gen_tensor_expr(var, info, recipe)])
    if getattr(recipe, "nonzero_fix", False):
        L.append("%s%s[%s == 0] = 1" % (indent, var, var))
    if getattr(recipe, "inject_seen", False):
        L.append("%sif %s.get(\"inject\"):" % (indent, info))
        L.append("%s    _f = %s.reshape(-1)" % (indent, var))
        L.append("%s    _f[0] = float(%s[\"inject\"])" % (indent, info))
        L.append("%s    %s = _f.reshape(%s.shape)" % (indent, var, var))
    return L


def extract_model_section(src):
    m = re.search(r"# ---- Model .*?----\n(.*?)\n# ---- prepare_inputs", src, re.S)
    if not m:
        raise ValueError("Model section markers not found")
    # device moves to NPU and back are value-preserving; drop them so the
    # reference runs on CPU (only CTCLossV3/CTCLossV3Grad contain these)
    return m.group(1).replace(".npu()", "").strip("\n")


def build_py(op_name, json_name, model_src, fwd_names, init_names,
             input_recipes, init_recipes, has_init):
    parts = [HEADER, model_src, "\n\n"]
    # get_input_groups
    parts.append("def get_input_groups():\n")
    parts.append("    json_path = os.path.join(os.path.dirname(__file__), '%s')\n" % json_name)
    parts.append("    with open(json_path, \"r\") as f:\n")
    parts.append("        cases = [json.loads(line) for line in f if line.strip()]\n\n")
    parts.append("    input_groups = []\n")
    parts.append("    for case in cases:\n")
    parts.append("        inputs = case[\"inputs\"]\n")
    for i, (nm, r) in enumerate(zip(fwd_names, input_recipes)):
        parts.append("        %s_info = inputs[%d]\n" % (nm, i))
    parts.append("\n")
    for nm, r in zip(fwd_names, input_recipes):
        for line in gen_arg_lines(nm, nm + "_info", r):
            parts.append(line + "\n")
        if r.kind == "tensor" or (r.kind == "mixed" and "tensor" in (r.mixed or {})):
            pass
    # requires_grad post-pass
    for nm, r in zip(fwd_names, input_recipes):
        if getattr(r, "requires_grad", False):
            if r.kind == "mixed":
                parts.append("        if isinstance(%s, torch.Tensor):\n" % nm)
                parts.append("            %s = %s.requires_grad_(True)\n" % (nm, nm))
            else:
                parts.append("        %s = %s.requires_grad_(True)\n" % (nm, nm))
    parts.append("\n        input_groups.append([%s])\n" % ", ".join(fwd_names))
    parts.append("    return input_groups\n\n\n")
    # get_init_inputs
    parts.append("def get_init_inputs():\n")
    if not has_init:
        parts.append("    return []\n")
    else:
        parts.append("    json_path = os.path.join(os.path.dirname(__file__), '%s')\n" % json_name)
        parts.append("    with open(json_path, \"r\") as f:\n")
        parts.append("        cases = [json.loads(line) for line in f if line.strip()]\n\n")
        parts.append("    init_groups = []\n")
        parts.append("    for case in cases:\n")
        parts.append("        entries = case.get(\"init_inputs\", [])\n")
        for i, nm in enumerate(init_names):
            parts.append("        %s_info = entries[%d]\n" % (nm, i))
        for nm, r in zip(init_names, init_recipes):
            for line in gen_arg_lines(nm, nm + "_info", r):
                parts.append(line + "\n")
        parts.append("        init_groups.append([%s])\n" % ", ".join(init_names))
        parts.append("    return init_groups\n")
    return "".join(parts)


# ---------------------------------------------------------------- main conversion

def sig_params(fn, skip_self=True):
    params = list(inspect.signature(fn).parameters.values())
    out = []
    for p in params:
        if skip_self and p.name == "self":
            continue
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        out.append(p.name)
    return out


def convert_op(level, old_id, op_name, src_py, new_id):
    tag = "%s_%d_%s" % (level, old_id, op_name)
    new_base = "cannops_%s_%d_%s" % (level, new_id, op_name)
    mod = load_module(src_py, tag, json_name="%d_%s.json" % (old_id, op_name))
    input_samples, init_groups = sample_op(mod)
    n_cases = len(input_samples[0])

    fwd_names = sig_params(mod.Model.forward)
    n_args = len(input_samples[0][0])
    if len(fwd_names) != n_args:
        fwd_names = ["arg%d" % i for i in range(n_args)]

    # detect args whose shape varies across samples of the same case (shape
    # derived from random values); informational, verify will catch breakage
    unstable = []
    for j in range(n_args):
        for c in range(n_cases):
            shp = {tuple(input_samples[s][c][j].shape) for s in range(SAMPLES)
                   if isinstance(input_samples[s][c][j], torch.Tensor)}
            if len(shp) > 1:
                unstable.append(j)
                break

    init_names = sig_params(mod.Model.__init__)
    has_init = len(init_names) > 0 and any(len(g) > 0 for g in init_groups)

    # classify per position
    input_recipes = []
    for j, nm in enumerate(fwd_names):
        by_case = [[input_samples[s][c][j] for s in range(SAMPLES)] for c in range(n_cases)]
        r = classify_position(nm, by_case)
        vals = [input_samples[s][c][j] for s in range(SAMPLES) for c in range(n_cases)]
        r.requires_grad = any(isinstance(v, torch.Tensor) and v.requires_grad for v in vals)
        if r.kind == "sparse":
            r.sparse_from = OP_OVERRIDES.get(op_name, {}).get("sparse_from")
        if nm in OP_OVERRIDES.get(op_name, {}).get("nonzero_fix_args", []):
            r.nonzero_fix = True
        # random attr: value differs across samples of the same case
        if r.kind == "attr" and r.dtype in ("int", "float"):
            rvals = [v for v in vals if isinstance(v, (int, float)) and not isinstance(v, bool)]
            if len({repr(v) for v in rvals}) > 1:
                per_case_same = all(
                    len({repr(input_samples[s][c][j]) for s in range(SAMPLES)}) == 1
                    for c in range(n_cases))
                if not per_case_same:
                    r.random = True
                    r.attr_range = [min(rvals), max(rvals)]
        input_recipes.append(r)

    init_recipes = []
    if has_init:
        n_init = len(init_groups[0])
        if len(init_names) != n_init:
            init_names = ["init%d" % i for i in range(n_init)]
        for j, nm in enumerate(init_names):
            by_case = [[init_groups[c][j]] for c in range(n_cases)]
            init_recipes.append(classify_position(nm, by_case))

    # json lines (use sample 0 concrete values + per-position ranges over all samples)
    force_names = set(OP_OVERRIDES.get(op_name, {}).get("force_value_args", []))

    def _force_data(entries, c):
        """Pin values for int/bool shape-driver args (when any arg has an
        unstable shape) and for override-named args, so recorded shapes stay
        consistent with regenerated inputs."""
        for j in range(n_args):
            if entries[j].get("type") != "tensor" or "data" in entries[j]:
                continue
            v = input_samples[0][c][j]
            if not isinstance(v, torch.Tensor) or v.numel() > FORCE_VALUE_MAX:
                continue
            is_driver = unstable and (v.dtype in INT_DTYPES or v.dtype == torch.bool)
            if is_driver or fwd_names[j] in force_names:
                entries[j] = {"name": entries[j]["name"], "type": "tensor",
                              "required": True, "dtype": DTYPE_NAME[v.dtype],
                              "shape": list(v.shape), "data": _tensor_value(v)}
                if input_recipes[j].requires_grad:
                    entries[j]["requires_grad"] = True

    lines = []
    for c in range(n_cases):
        entries = []
        for j, r in enumerate(input_recipes):
            vals = [input_samples[s][c][j] for s in range(SAMPLES)]
            entries.append(serialize_entry(r, vals, r.requires_grad))
        for j in unstable:
            if entries[j].get("type") == "tensor":
                entries[j]["shape_derived"] = True  # shape depends on random driver values
        _force_data(entries, c)
        case_obj = {"inputs": entries}
        if has_init:
            init_entries = []
            for j, r in enumerate(init_recipes):
                init_entries.append(serialize_entry(r, [init_groups[c][j]], False))
            case_obj["init_inputs"] = init_entries
        lines.append(json.dumps(case_obj, ensure_ascii=False))

    src = open(src_py).read()
    model_src = extract_model_section(src)
    for old_t, new_t in OP_OVERRIDES.get(op_name, {}).get("model_patch", []):
        model_src = model_src.replace(old_t, new_t)
    py_text = build_py(op_name, new_base + ".json", model_src, fwd_names, init_names,
                       input_recipes, init_recipes, has_init)

    with open(os.path.join(OUT_DIR, new_base + ".json"), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(OUT_DIR, new_base + ".py"), "w") as f:
        f.write(py_text)

    tiers = {"randn": 0, "rand": 0, "attr": 0}
    del mod, input_samples, init_groups
    gc.collect()
    return new_base, {"unstable_args": unstable}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    manifest = []
    for level in ("level1", "level2", "level3"):
        lvl_dir = os.path.join(SRC_DIR, level)
        ops = []
        for fn in os.listdir(lvl_dir):
            m = re.match(r"^(\d+)_(.+)\.py$", fn)
            if m:
                ops.append((int(m.group(1)), m.group(2), os.path.join(lvl_dir, fn)))
        ops.sort()
        for new_id, (old_id, op_name, src_py) in enumerate(ops):
            try:
                new_base, info = convert_op(level, old_id, op_name, src_py, new_id)
                manifest.append({"level": level, "old_id": old_id, "op": op_name,
                                 "new": new_base, "status": "ok", **info})
                print("[ok] %s/%d_%s -> %s" % (level, old_id, op_name, new_base), flush=True)
            except Exception as e:
                manifest.append({"level": level, "old_id": old_id, "op": op_name,
                                 "status": "fail", "error": repr(e)})
                print("[FAIL] %s/%d_%s: %r" % (level, old_id, op_name, e), flush=True)
    with open(os.path.join(OUT_DIR, "_manifest.json"), "w") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    n_ok = sum(1 for m in manifest if m["status"] == "ok")
    print("converted %d/%d" % (n_ok, len(manifest)))


if __name__ == "__main__":
    main()
