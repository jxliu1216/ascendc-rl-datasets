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
CPU 金标准与 ops-transformer `executor_aclnnDequantRopeQuantKvcache` 中 drqk 逻辑对齐：
dequant(可选) → split → RoPE（与内核一致：对 sin 前半取负后与 half-swap 组合）→ 量化并 scatter 到 k/v cache。
"""
from typing import List, Optional, Sequence, Tuple
import torch
import torch.nn as nn
QUANT_MIN = -128
QUANT_MAX = 127

def _rope_match_kernel(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """与 AscendC 实现一致：对 sin 前半轴取负，尾维后半与前半交换后与 sin 组合，再与 x*cos 相加（见 op_kernel 中 RoPE 段）。"""
    d = x.shape[-1]
    half = d // 2
    sin_adj = sin.clone()
    sin_adj[..., :half] = -sin_adj[..., :half]
    x_rot = torch.cat((x[..., half:], x[..., :half]), dim=-1)
    return x * cos + x_rot * sin_adj

def _dequant_like_executor(inp: torch.Tensor, bias: Optional[torch.Tensor], weight_scale: Optional[torch.Tensor], activation_scale: Optional[torch.Tensor]) -> torch.Tensor:
    if weight_scale is None:
        return inp
    ws = weight_scale.reshape(1, 1, -1)
    if activation_scale is not None:
        as_cpu = activation_scale.reshape(inp.shape[0], inp.shape[1], 1)
    else:
        as_cpu = None
    if bias is not None:
        bias_cpu = bias.reshape(1, 1, -1)
        if bias.dtype == torch.int32:
            t = torch.add(inp, bias_cpu)
            t = torch.mul(t, ws)
            if as_cpu is not None:
                t = torch.mul(t, as_cpu)
        else:
            t = torch.mul(inp, ws)
            if as_cpu is not None:
                t = torch.mul(t, as_cpu)
            t = torch.add(t, bias_cpu)
    else:
        t = torch.mul(inp, ws)
        if as_cpu is not None:
            t = torch.mul(t, as_cpu)
    return t

def _quant_update_scatter(key_cache: torch.Tensor, key: torch.Tensor, inv_scale: torch.Tensor, indice: torch.Tensor, offset: Optional[torch.Tensor], page_mode: bool) -> None:
    scale = inv_scale.reshape(-1, key.shape[-1])
    off = offset.reshape(-1, key.shape[-1]) if offset is not None else None
    if off is not None:
        quant_out = key.float() * scale + off
    else:
        quant_out = key.float() * scale
    quant_out = torch.round(quant_out)
    quant_out1 = torch.clamp(torch.round(quant_out.float()), min=QUANT_MIN, max=QUANT_MAX).to(torch.int8)
    if page_mode:
        d0, d1, d2, d3 = key_cache.shape
        key_cache_pa = key_cache.reshape(-1, key_cache.shape[-2], key_cache.shape[-1])
        quant_out2 = quant_out1.reshape(-1, quant_out1.shape[-2], quant_out1.shape[-1])
        for b in range(indice.shape[0]):
            iv = int(indice[b].item())
            key_cache_pa[iv] = quant_out2[b]
        key_cache.copy_(key_cache_pa.reshape(d0, d1, d2, d3))
    else:
        s_len = quant_out1.shape[1]
        for b in range(indice.shape[0]):
            iv = int(indice[b].item())
            key_cache[b, iv:iv + s_len, :, :].copy_(quant_out1[b])

def _run_drqk(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, indices: torch.Tensor, scale_k: torch.Tensor, scale_v: torch.Tensor, size_splits: Sequence[int], offset_k: Optional[torch.Tensor], offset_v: Optional[torch.Tensor], weight_scale: Optional[torch.Tensor], activation_scale: Optional[torch.Tensor], bias: Optional[torch.Tensor], cache_mode: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    page_mode = str(cache_mode).lower() == 'page'
    x_work = x
    if x.dtype == torch.int32:
        x_work = _dequant_like_executor(x.float(), bias, weight_scale, activation_scale)
        x_work = x_work.to(cos.dtype)
    else:
        x_work = x
    if x_work.dim() == 2:
        x_work = x_work.unsqueeze(1)
    cos_u, sin_u = (cos, sin)
    if cos_u.dim() == 2:
        cos_u = cos_u.unsqueeze(1).unsqueeze(2)
        sin_u = sin_u.unsqueeze(1).unsqueeze(2)
    b, s, _ = x_work.shape
    h = k_cache.shape[-1]
    q, kt, vt = x_work.split(tuple(size_splits), dim=-1)
    q1 = q.reshape(b, s, -1, h)
    k1 = kt.reshape(b, s, -1, h)
    v1 = vt.reshape(b, s, -1, h)
    ropek = _rope_match_kernel(k1, cos_u, sin_u)
    ropeq = _rope_match_kernel(q1, cos_u, sin_u)
    inv_k = (1.0 / scale_k.to(torch.float32)).to(torch.float32)
    inv_v = (1.0 / scale_v.to(torch.float32)).to(torch.float32)
    kc = k_cache.clone()
    vc = v_cache.clone()
    _quant_update_scatter(kc, ropek, inv_k, indices, offset_k, page_mode)
    _quant_update_scatter(vc, v1, inv_v, indices, offset_v, page_mode)
    return (ropeq, ropek, v1, kc, vc)

class Model(nn.Module):

    def __init__(self, size_splits: Tuple[int, int, int], kv_output: bool=True, quant_mode: str='static', layout: str='BSND', cache_mode: str='contiguous'):
        super().__init__()
        self.size_splits = size_splits
        self.kv_output = kv_output
        self.quant_mode = quant_mode
        self.layout = layout
        self.cache_mode = cache_mode

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, indices: torch.Tensor, scale_k: torch.Tensor, scale_v: torch.Tensor, offset_k: Optional[torch.Tensor], offset_v: Optional[torch.Tensor], weight_scale: Optional[torch.Tensor], activation_scale: Optional[torch.Tensor], bias: Optional[torch.Tensor], _q_buf: torch.Tensor, _k_buf: torch.Tensor, _v_buf: torch.Tensor) -> List[torch.Tensor]:
        del _q_buf, _k_buf, _v_buf
        ropeq, ropek, v1, kco, vco = _run_drqk(x, cos, sin, k_cache, v_cache, indices, scale_k, scale_v, self.size_splits, offset_k, offset_v, weight_scale, activation_scale, bias, self.cache_mode)
        return [ropeq, ropek, v1, kco, vco]

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level3_3_DequantRopeQuantKvcache.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        x_info = inputs[0]
        cos_info = inputs[1]
        sin_info = inputs[2]
        k_cache_info = inputs[3]
        v_cache_info = inputs[4]
        indices_info = inputs[5]
        scale_k_info = inputs[6]
        scale_v_info = inputs[7]
        offset_k_info = inputs[8]
        offset_v_info = inputs[9]
        weight_scale_info = inputs[10]
        activation_scale_info = inputs[11]
        bias_info = inputs[12]
        _q_buf_info = inputs[13]
        _k_buf_info = inputs[14]
        _v_buf_info = inputs[15]

        if "data" in x_info:
            x = torch.tensor(x_info["data"], dtype=DTYPE_MAP[x_info["dtype"]]).reshape(x_info["shape"])
        else:
            _dt = DTYPE_MAP[x_info["dtype"]]
            if _dt in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
                x = torch.randint(x_info["range"][0], x_info["range"][1] + 1, tuple(x_info["shape"]), dtype=_dt)
            elif _dt == torch.bool:
                x = torch.rand(x_info["shape"]) < x_info.get("true_frac", 0.5)
            else:
                x = torch.rand(x_info["shape"], dtype=_dt) * (x_info["range"][1] - x_info["range"][0]) + x_info["range"][0]
        if "data" in cos_info:
            cos = torch.tensor(cos_info["data"], dtype=DTYPE_MAP[cos_info["dtype"]]).reshape(cos_info["shape"])
        else:
            cos = torch.rand(cos_info["shape"], dtype=DTYPE_MAP[cos_info["dtype"]]) * (cos_info["range"][1] - cos_info["range"][0]) + cos_info["range"][0]
        if "data" in sin_info:
            sin = torch.tensor(sin_info["data"], dtype=DTYPE_MAP[sin_info["dtype"]]).reshape(sin_info["shape"])
        else:
            sin = torch.rand(sin_info["shape"], dtype=DTYPE_MAP[sin_info["dtype"]]) * (sin_info["range"][1] - sin_info["range"][0]) + sin_info["range"][0]
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
        if "data" in scale_k_info:
            scale_k = torch.tensor(scale_k_info["data"], dtype=DTYPE_MAP[scale_k_info["dtype"]]).reshape(scale_k_info["shape"])
        else:
            scale_k = torch.rand(scale_k_info["shape"], dtype=DTYPE_MAP[scale_k_info["dtype"]]) * (scale_k_info["range"][1] - scale_k_info["range"][0]) + scale_k_info["range"][0]
        if "data" in scale_v_info:
            scale_v = torch.tensor(scale_v_info["data"], dtype=DTYPE_MAP[scale_v_info["dtype"]]).reshape(scale_v_info["shape"])
        else:
            scale_v = torch.rand(scale_v_info["shape"], dtype=DTYPE_MAP[scale_v_info["dtype"]]) * (scale_v_info["range"][1] - scale_v_info["range"][0]) + scale_v_info["range"][0]
        if offset_k_info["type"] == "attr":
            if offset_k_info.get("dtype") == "none":
                offset_k = None
            else:
                offset_k = offset_k_info["value"]
        else:
            if "data" in offset_k_info:
                offset_k = torch.tensor(offset_k_info["data"], dtype=DTYPE_MAP[offset_k_info["dtype"]]).reshape(offset_k_info["shape"])
            else:
                offset_k = torch.randn(offset_k_info["shape"], dtype=DTYPE_MAP[offset_k_info["dtype"]]) * offset_k_info["std"] + offset_k_info["mean"]
        if offset_v_info["type"] == "attr":
            if offset_v_info.get("dtype") == "none":
                offset_v = None
            else:
                offset_v = offset_v_info["value"]
        else:
            if "data" in offset_v_info:
                offset_v = torch.tensor(offset_v_info["data"], dtype=DTYPE_MAP[offset_v_info["dtype"]]).reshape(offset_v_info["shape"])
            else:
                offset_v = torch.randn(offset_v_info["shape"], dtype=DTYPE_MAP[offset_v_info["dtype"]]) * offset_v_info["std"] + offset_v_info["mean"]
        if weight_scale_info["type"] == "attr":
            if weight_scale_info.get("dtype") == "none":
                weight_scale = None
            else:
                weight_scale = weight_scale_info["value"]
        else:
            if "data" in weight_scale_info:
                weight_scale = torch.tensor(weight_scale_info["data"], dtype=DTYPE_MAP[weight_scale_info["dtype"]]).reshape(weight_scale_info["shape"])
            else:
                weight_scale = torch.rand(weight_scale_info["shape"], dtype=DTYPE_MAP[weight_scale_info["dtype"]]) * (weight_scale_info["range"][1] - weight_scale_info["range"][0]) + weight_scale_info["range"][0]
        if activation_scale_info["type"] == "attr":
            if activation_scale_info.get("dtype") == "none":
                activation_scale = None
            else:
                activation_scale = activation_scale_info["value"]
        else:
            if "data" in activation_scale_info:
                activation_scale = torch.tensor(activation_scale_info["data"], dtype=DTYPE_MAP[activation_scale_info["dtype"]]).reshape(activation_scale_info["shape"])
            else:
                activation_scale = torch.rand(activation_scale_info["shape"], dtype=DTYPE_MAP[activation_scale_info["dtype"]])
        if bias_info["type"] == "attr":
            if bias_info.get("dtype") == "none":
                bias = None
            else:
                bias = bias_info["value"]
        else:
            if "data" in bias_info:
                bias = torch.tensor(bias_info["data"], dtype=DTYPE_MAP[bias_info["dtype"]]).reshape(bias_info["shape"])
            else:
                _dt = DTYPE_MAP[bias_info["dtype"]]
                if _dt in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
                    bias = torch.randint(bias_info["range"][0], bias_info["range"][1] + 1, tuple(bias_info["shape"]), dtype=_dt)
                elif _dt == torch.bool:
                    bias = torch.rand(bias_info["shape"]) < bias_info.get("true_frac", 0.5)
                else:
                    bias = torch.randn(bias_info["shape"], dtype=_dt) * bias_info["std"] + bias_info["mean"]
        _q_buf = torch.empty(_q_buf_info["shape"], dtype=DTYPE_MAP[_q_buf_info["dtype"]])
        _k_buf = torch.empty(_k_buf_info["shape"], dtype=DTYPE_MAP[_k_buf_info["dtype"]])
        _v_buf = torch.empty(_v_buf_info["shape"], dtype=DTYPE_MAP[_v_buf_info["dtype"]])

        input_groups.append([x, cos, sin, k_cache, v_cache, indices, scale_k, scale_v, offset_k, offset_v, weight_scale, activation_scale, bias, _q_buf, _k_buf, _v_buf])
    return input_groups


def get_init_inputs():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level3_3_DequantRopeQuantKvcache.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    init_groups = []
    for case in cases:
        entries = case.get("init_inputs", [])
        size_splits_info = entries[0]
        kv_output_info = entries[1]
        quant_mode_info = entries[2]
        layout_info = entries[3]
        cache_mode_info = entries[4]
        size_splits = tuple(size_splits_info["value"])
        kv_output = kv_output_info["value"]
        quant_mode = quant_mode_info["value"]
        layout = layout_info["value"]
        cache_mode = cache_mode_info["value"]
        init_groups.append([size_splits, kv_output, quant_mode, layout, cache_mode])
    return init_groups
