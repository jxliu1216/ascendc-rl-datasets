import torch
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
CPU 金标准：qkv 切分 → 对 q/k 做与 AscendC 一致的 half RoPE → k/v 按 per-D 的 scale/offset 量化并写入 cache。
内核未将 RoPE 后的 k 写回 k_out GM，精度验证仅比对 q（见 prepare_inputs.custom_check_precision）。
"""
from typing import List, Sequence, Tuple
import torch
import torch.nn as nn
QUANT_MIN = -128
QUANT_MAX = 127

def _rope_match_kernel(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1]
    half = d // 2
    sin_adj = sin.clone()
    sin_adj[..., :half] = -sin_adj[..., :half]
    x_rot = torch.cat((x[..., half:], x[..., :half]), dim=-1)
    return x * cos + x_rot * sin_adj

def _quant_to_cache(x_f16: torch.Tensor, quant_scale: torch.Tensor, quant_offset: torch.Tensor) -> torch.Tensor:
    scale = quant_scale.to(torch.float32).reshape(1, 1, 1, -1)
    off = quant_offset.to(torch.float32).reshape(1, 1, 1, -1)
    t = x_f16.float() / scale + off
    t = torch.round(t)
    return torch.clamp(t, min=QUANT_MIN, max=QUANT_MAX).to(torch.int8)

def _scatter_cache(cache: torch.Tensor, quantized: torch.Tensor, indices: torch.Tensor) -> None:
    b = quantized.shape[0]
    s = quantized.shape[1]
    for bi in range(b):
        iv = int(indices[bi].item())
        cache[bi, iv:iv + s, :, :].copy_(quantized[bi])

def _run_rope_quant(qkv: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, indices: torch.Tensor, quant_scale: torch.Tensor, quant_offset: torch.Tensor, size_splits: Sequence[int]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    b, s, _ = qkv.shape
    sq, sk, sv = size_splits
    d = quant_scale.numel()
    n_q = sq // d
    n_kv = sk // d
    assert sq == n_q * d and sk == n_kv * d and (sv == n_kv * d)
    q_flat, k_flat, v_flat = qkv.split([sq, sk, sv], dim=-1)
    q_b = q_flat.reshape(b, s, n_q, d)
    k_b = k_flat.reshape(b, s, n_kv, d)
    v_b = v_flat.reshape(b, s, n_kv, d)
    cos_b = torch.broadcast_to(cos.float(), q_b.shape)
    sin_b = torch.broadcast_to(sin.float(), q_b.shape)
    cos_k = torch.broadcast_to(cos.float(), k_b.shape)
    sin_k = torch.broadcast_to(sin.float(), k_b.shape)
    rope_q = _rope_match_kernel(q_b.float(), cos_b, sin_b).to(torch.float16)
    rope_k = _rope_match_kernel(k_b.float(), cos_k, sin_k).to(torch.float16)
    k_q = _quant_to_cache(rope_k, quant_scale, quant_offset)
    v_q = _quant_to_cache(v_b, quant_scale, quant_offset)
    kc = k_cache.clone()
    vc = v_cache.clone()
    _scatter_cache(kc, k_q, indices)
    _scatter_cache(vc, v_q, indices)
    return (rope_q, rope_k, v_b, kc, vc)

class Model(nn.Module):

    def __init__(self, size_splits: Tuple[int, int, int], kv_output: bool=True, layout: str='BSND'):
        super().__init__()
        self.size_splits = tuple(size_splits)
        self.kv_output = kv_output
        self.layout = layout

    def forward(self, qkv: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, quant_scale: torch.Tensor, quant_offset: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, indices: torch.Tensor, _q_buf: torch.Tensor, _k_buf: torch.Tensor, _v_buf: torch.Tensor) -> List[torch.Tensor]:
        del _q_buf, _k_buf, _v_buf
        return list(_run_rope_quant(qkv, cos, sin, k_cache, v_cache, indices, quant_scale, quant_offset, self.size_splits))

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level3_42_RopeQuantKvcache.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        qkv_info = inputs[0]
        cos_info = inputs[1]
        sin_info = inputs[2]
        quant_scale_info = inputs[3]
        quant_offset_info = inputs[4]
        k_cache_info = inputs[5]
        v_cache_info = inputs[6]
        indices_info = inputs[7]
        _q_buf_info = inputs[8]
        _k_buf_info = inputs[9]
        _v_buf_info = inputs[10]

        if "data" in qkv_info:
            qkv = torch.tensor(qkv_info["data"], dtype=DTYPE_MAP[qkv_info["dtype"]]).reshape(qkv_info["shape"])
        else:
            qkv = torch.rand(qkv_info["shape"], dtype=DTYPE_MAP[qkv_info["dtype"]]) * (qkv_info["range"][1] - qkv_info["range"][0]) + qkv_info["range"][0]
        if "data" in cos_info:
            cos = torch.tensor(cos_info["data"], dtype=DTYPE_MAP[cos_info["dtype"]]).reshape(cos_info["shape"])
        else:
            cos = torch.rand(cos_info["shape"], dtype=DTYPE_MAP[cos_info["dtype"]]) * (cos_info["range"][1] - cos_info["range"][0]) + cos_info["range"][0]
        if "data" in sin_info:
            sin = torch.tensor(sin_info["data"], dtype=DTYPE_MAP[sin_info["dtype"]]).reshape(sin_info["shape"])
        else:
            sin = torch.rand(sin_info["shape"], dtype=DTYPE_MAP[sin_info["dtype"]]) * (sin_info["range"][1] - sin_info["range"][0]) + sin_info["range"][0]
        if "data" in quant_scale_info:
            quant_scale = torch.tensor(quant_scale_info["data"], dtype=DTYPE_MAP[quant_scale_info["dtype"]]).reshape(quant_scale_info["shape"])
        else:
            quant_scale = torch.rand(quant_scale_info["shape"], dtype=DTYPE_MAP[quant_scale_info["dtype"]])
        if "data" in quant_offset_info:
            quant_offset = torch.tensor(quant_offset_info["data"], dtype=DTYPE_MAP[quant_offset_info["dtype"]]).reshape(quant_offset_info["shape"])
        else:
            quant_offset = torch.randint(quant_offset_info["range"][0], quant_offset_info["range"][1] + 1, tuple(quant_offset_info["shape"]), dtype=DTYPE_MAP[quant_offset_info["dtype"]])
        if "data" in k_cache_info:
            k_cache = torch.tensor(k_cache_info["data"], dtype=DTYPE_MAP[k_cache_info["dtype"]]).reshape(k_cache_info["shape"])
        else:
            k_cache = torch.randint(k_cache_info["range"][0], k_cache_info["range"][1] + 1, tuple(k_cache_info["shape"]), dtype=DTYPE_MAP[k_cache_info["dtype"]])
        if "data" in v_cache_info:
            v_cache = torch.tensor(v_cache_info["data"], dtype=DTYPE_MAP[v_cache_info["dtype"]]).reshape(v_cache_info["shape"])
        else:
            v_cache = torch.randint(v_cache_info["range"][0], v_cache_info["range"][1] + 1, tuple(v_cache_info["shape"]), dtype=DTYPE_MAP[v_cache_info["dtype"]])
        if "data" in indices_info:
            indices = torch.tensor(indices_info["data"], dtype=DTYPE_MAP[indices_info["dtype"]]).reshape(indices_info["shape"])
        else:
            indices = torch.randint(indices_info["range"][0], indices_info["range"][1] + 1, tuple(indices_info["shape"]), dtype=DTYPE_MAP[indices_info["dtype"]])
        if "data" in _q_buf_info:
            _q_buf = torch.tensor(_q_buf_info["data"], dtype=DTYPE_MAP[_q_buf_info["dtype"]]).reshape(_q_buf_info["shape"])
        else:
            _q_buf = torch.randn(_q_buf_info["shape"], dtype=DTYPE_MAP[_q_buf_info["dtype"]]) * _q_buf_info["std"] + _q_buf_info["mean"]
        if _q_buf_info.get("inject"):
            _f = _q_buf.reshape(-1)
            _f[0] = float(_q_buf_info["inject"])
            _q_buf = _f.reshape(_q_buf.shape)
        if "data" in _k_buf_info:
            _k_buf = torch.tensor(_k_buf_info["data"], dtype=DTYPE_MAP[_k_buf_info["dtype"]]).reshape(_k_buf_info["shape"])
        else:
            _k_buf = torch.randn(_k_buf_info["shape"], dtype=DTYPE_MAP[_k_buf_info["dtype"]]) * _k_buf_info["std"] + _k_buf_info["mean"]
        if _k_buf_info.get("inject"):
            _f = _k_buf.reshape(-1)
            _f[0] = float(_k_buf_info["inject"])
            _k_buf = _f.reshape(_k_buf.shape)
        if "data" in _v_buf_info:
            _v_buf = torch.tensor(_v_buf_info["data"], dtype=DTYPE_MAP[_v_buf_info["dtype"]]).reshape(_v_buf_info["shape"])
        else:
            _v_buf = torch.randn(_v_buf_info["shape"], dtype=DTYPE_MAP[_v_buf_info["dtype"]]) * _v_buf_info["std"] + _v_buf_info["mean"]
        if _v_buf_info.get("inject"):
            _f = _v_buf.reshape(-1)
            _f[0] = float(_v_buf_info["inject"])
            _v_buf = _f.reshape(_v_buf.shape)

        input_groups.append([qkv, cos, sin, quant_scale, quant_offset, k_cache, v_cache, indices, _q_buf, _k_buf, _v_buf])
    return input_groups


def get_init_inputs():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level3_42_RopeQuantKvcache.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    init_groups = []
    for case in cases:
        entries = case.get("init_inputs", [])
        size_splits_info = entries[0]
        kv_output_info = entries[1]
        layout_info = entries[2]
        size_splits = tuple(size_splits_info["value"])
        kv_output = kv_output_info["value"]
        layout = layout_info["value"]
        init_groups.append([size_splits, kv_output, layout])
    return init_groups
