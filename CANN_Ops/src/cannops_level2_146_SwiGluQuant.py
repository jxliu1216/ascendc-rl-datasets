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
CPU 金标准对齐 kernel：SwiGLU 与文档非 MoE / MoE 动态公式；动态量化 scale = dstScale / rowmax(|Ytmp|)；
Cast 路径近似 swi_glu_quant_base.h::CastQuantOut（rint→int32→fp16→trunc）。
静态量化 scale 输出在核上写 0，golden 对 scale 全 0 比对。
"""
from typing import List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
DT_INT8 = 2
DT_INT4 = 29

def swiglu_act(x_f: torch.Tensor, activate_left: bool) -> torch.Tensor:
    h = x_f.shape[-1] // 2
    a = x_f[..., :h]
    b = x_f[..., h:]
    if activate_left:
        return torch.nn.functional.silu(a) * b
    return torch.nn.functional.silu(b) * a

def cast_like_kernel(fp: torch.Tensor, dst_type: int) -> torch.Tensor:
    t = fp.detach().float()
    i32 = torch.round(t).to(torch.int32)
    h = i32.to(torch.float16).float()
    if int(dst_type) == DT_INT4:
        out = torch.round(h).clamp(-8, 7).to(torch.int8)
    else:
        out = torch.trunc(h).clamp(-128, 127).to(torch.int8)
    return out.to(device=fp.device)

def _dynamic_segment(seg: torch.Tensor, dst_scale: float) -> Tuple[torch.Tensor, torch.Tensor]:
    m = seg.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
    s = dst_scale / m
    q = seg * s
    return (q, s.squeeze(-1))

def golden_dynamic(x: torch.Tensor, smooth: torch.Tensor, activate_left: bool, dst_type: int, group_index: Optional[torch.Tensor], group_list_type: int) -> Tuple[torch.Tensor, torch.Tensor]:
    act = swiglu_act(x.float(), activate_left)
    dst_scale = 127.0 if int(dst_type) == DT_INT8 else 7.0
    sm = smooth.float()
    lead = act.shape[:-1]
    rn = int(torch.tensor(lead, dtype=torch.int64).prod().item())
    nc = act.shape[-1]
    a2 = act.reshape(rn, nc)
    y_acc = torch.zeros(rn, nc, dtype=torch.float32, device=x.device)
    scale_1d = torch.zeros(rn, dtype=torch.float32, device=x.device)
    if group_index is None:
        sm2 = sm.unsqueeze(0) if sm.dim() == 1 else sm
        if sm2.shape[-1] != nc:
            raise ValueError('smooth last dim must match SwiGLU output width')
        y_tmp = a2 * sm2
        q, sc = _dynamic_segment(y_tmp, dst_scale)
        y_acc = q
        scale_1d = sc
    else:
        g = group_index.detach().cpu().numpy().astype(np.int64)
        if int(group_list_type) == 1:
            g = np.cumsum(g)
        G = int(sm.shape[0])
        sm2 = sm.reshape(G, -1)
        if sm2.shape[1] != nc:
            raise ValueError('MoE smooth shape mismatch')
        start = 0
        for gi in range(len(g)):
            end = int(g[gi])
            if end <= start or end > rn:
                continue
            seg = a2[start:end] * sm2[gi:gi + 1]
            q, sc = _dynamic_segment(seg, dst_scale)
            y_acc[start:end] = q
            scale_1d[start:end] = sc
            start = end
    y_q = cast_like_kernel(y_acc.reshape_as(act), dst_type)
    scale_out = scale_1d.reshape(lead)
    return (y_q, scale_out)

def golden_static(x: torch.Tensor, smooth: torch.Tensor, offset: torch.Tensor, activate_left: bool, dst_type: int, group_index: Optional[torch.Tensor], group_list_type: int, static_mode: str) -> Tuple[torch.Tensor, torch.Tensor]:
    act = swiglu_act(x.float(), activate_left)
    sm = smooth.float()
    off = offset.float()
    lead = act.shape[:-1]
    rn = int(torch.tensor(lead, dtype=torch.int64).prod().item())
    nc = act.shape[-1]
    a2 = act.reshape(rn, nc)
    if group_index is None:
        if static_mode == 'per_tensor':
            y_tmp = a2 * sm.reshape(-1)[0] + off.reshape(-1)[0]
        else:
            sm2 = sm.unsqueeze(0) if sm.dim() == 1 else sm
            off2 = off.unsqueeze(0) if off.dim() == 1 else off
            y_tmp = a2 * sm2 + off2
    else:
        g = group_index.detach().cpu().numpy().astype(np.int64)
        if int(group_list_type) == 1:
            g = np.cumsum(g)
        G = int(sm.shape[0])
        sm2 = sm.reshape(G, -1) if sm.dim() > 1 else sm.reshape(G, 1)
        off2 = off.reshape(G, -1) if off.dim() > 1 else off.reshape(G, 1)
        y_acc = torch.zeros(rn, nc, dtype=torch.float32, device=x.device)
        start = 0
        for gi in range(len(g)):
            end = int(g[gi])
            if end <= start or end > rn:
                continue
            if static_mode == 'per_tensor':
                y_acc[start:end] = a2[start:end] * sm2[gi, 0] + off2[gi, 0]
            else:
                y_acc[start:end] = a2[start:end] * sm2[gi:gi + 1] + off2[gi:gi + 1]
            start = end
        y_tmp = y_acc
    y_q = cast_like_kernel(y_tmp.reshape_as(act), dst_type)
    scale_z = torch.zeros(lead, dtype=torch.float32, device=x.device)
    return (y_q, scale_z)

class Model(nn.Module):

    def __init__(self, activate_left: bool, quant_mode: str, group_list_type: int, dst_type: int, static_mode: str):
        super().__init__()
        self.activate_left = bool(activate_left)
        self.quant_mode = str(quant_mode)
        self.group_list_type = int(group_list_type)
        self.dst_type = int(dst_type)
        self.static_mode = str(static_mode)

    def forward(self, x: torch.Tensor, smooth: torch.Tensor, offset: Optional[torch.Tensor], group_index: Optional[torch.Tensor]) -> List[torch.Tensor]:
        if self.quant_mode == 'dynamic':
            y, s = golden_dynamic(x, smooth, self.activate_left, self.dst_type, group_index, self.group_list_type)
        else:
            assert offset is not None
            y, s = golden_static(x, smooth, offset, self.activate_left, self.dst_type, group_index, self.group_list_type, self.static_mode)
        return [y, s]

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level2_146_SwiGluQuant.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        x_info = inputs[0]
        smooth_info = inputs[1]
        offset_info = inputs[2]
        group_index_info = inputs[3]

        if "data" in x_info:
            x = torch.tensor(x_info["data"], dtype=DTYPE_MAP[x_info["dtype"]]).reshape(x_info["shape"])
        else:
            x = torch.rand(x_info["shape"], dtype=DTYPE_MAP[x_info["dtype"]]) * (x_info["range"][1] - x_info["range"][0]) + x_info["range"][0]
        if "data" in smooth_info:
            smooth = torch.tensor(smooth_info["data"], dtype=DTYPE_MAP[smooth_info["dtype"]]).reshape(smooth_info["shape"])
        else:
            smooth = torch.rand(smooth_info["shape"], dtype=DTYPE_MAP[smooth_info["dtype"]])
        if offset_info["type"] == "attr":
            if offset_info.get("dtype") == "none":
                offset = None
            else:
                offset = offset_info["value"]
        else:
            if "data" in offset_info:
                offset = torch.tensor(offset_info["data"], dtype=DTYPE_MAP[offset_info["dtype"]]).reshape(offset_info["shape"])
            else:
                offset = torch.rand(offset_info["shape"], dtype=DTYPE_MAP[offset_info["dtype"]])
        if group_index_info["type"] == "attr":
            if group_index_info.get("dtype") == "none":
                group_index = None
            else:
                group_index = group_index_info["value"]
        else:
            if "data" in group_index_info:
                group_index = torch.tensor(group_index_info["data"], dtype=DTYPE_MAP[group_index_info["dtype"]]).reshape(group_index_info["shape"])
            else:
                group_index = torch.randint(group_index_info["range"][0], group_index_info["range"][1] + 1, tuple(group_index_info["shape"]), dtype=DTYPE_MAP[group_index_info["dtype"]])

        input_groups.append([x, smooth, offset, group_index])
    return input_groups


def get_init_inputs():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level2_146_SwiGluQuant.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    init_groups = []
    for case in cases:
        entries = case.get("init_inputs", [])
        activate_left_info = entries[0]
        quant_mode_info = entries[1]
        group_list_type_info = entries[2]
        dst_type_info = entries[3]
        static_mode_info = entries[4]
        activate_left = activate_left_info["value"]
        quant_mode = quant_mode_info["value"]
        group_list_type = group_list_type_info["value"]
        dst_type = dst_type_info["value"]
        static_mode = static_mode_info["value"]
        init_groups.append([activate_left, quant_mode, group_list_type, dst_type, static_mode])
    return init_groups
