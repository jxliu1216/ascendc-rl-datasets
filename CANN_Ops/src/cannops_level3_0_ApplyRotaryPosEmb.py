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
CPU 金标准与 ascendc ApplyRotaryPosEmb（half rotary_mode，与 aclnn 默认一致）对齐：
对最后一维前半 sin 取负，将 x 后半与前半交换后与 sin' 相乘，再与 x*cos 相加（见 apply_rotary_pos_emb_small.h::ComputeTotary）。
"""
from typing import List, Tuple
import torch
import torch.nn as nn

def golden_apply_rotary_pos_emb_half(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """与内核 half 模式一致；在 float32 上计算再 cast 回输入 dtype。"""
    od = q.dtype
    qf = q.float()
    kf = k.float()
    cos_b = torch.broadcast_to(cos.float(), qf.shape)
    sin_b = torch.broadcast_to(sin.float(), qf.shape)
    d = qf.shape[-1]
    half = d // 2
    sin_neg = sin_b.clone()
    sin_neg[..., :half] = -sin_neg[..., :half]

    def one(x: torch.Tensor) -> torch.Tensor:
        x_rot = torch.cat([x[..., half:], x[..., :half]], dim=-1)
        return x * cos_b + x_rot * sin_neg
    qo = one(qf).to(od)
    ko = one(kf).to(od)
    return (qo, ko)

class Model(nn.Module):

    def __init__(self, layout: int=1):
        super().__init__()
        self.layout = layout

    def forward(self, q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> List[torch.Tensor]:
        qo, ko = golden_apply_rotary_pos_emb_half(q, k, cos, sin)
        return [qo, ko]

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level3_0_ApplyRotaryPosEmb.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        q_info = inputs[0]
        k_info = inputs[1]
        cos_info = inputs[2]
        sin_info = inputs[3]

        if "data" in q_info:
            q = torch.tensor(q_info["data"], dtype=DTYPE_MAP[q_info["dtype"]]).reshape(q_info["shape"])
        else:
            q = torch.rand(q_info["shape"], dtype=DTYPE_MAP[q_info["dtype"]]) * (q_info["range"][1] - q_info["range"][0]) + q_info["range"][0]
        if "data" in k_info:
            k = torch.tensor(k_info["data"], dtype=DTYPE_MAP[k_info["dtype"]]).reshape(k_info["shape"])
        else:
            k = torch.rand(k_info["shape"], dtype=DTYPE_MAP[k_info["dtype"]]) * (k_info["range"][1] - k_info["range"][0]) + k_info["range"][0]
        if "data" in cos_info:
            cos = torch.tensor(cos_info["data"], dtype=DTYPE_MAP[cos_info["dtype"]]).reshape(cos_info["shape"])
        else:
            cos = torch.rand(cos_info["shape"], dtype=DTYPE_MAP[cos_info["dtype"]]) * (cos_info["range"][1] - cos_info["range"][0]) + cos_info["range"][0]
        if "data" in sin_info:
            sin = torch.tensor(sin_info["data"], dtype=DTYPE_MAP[sin_info["dtype"]]).reshape(sin_info["shape"])
        else:
            sin = torch.rand(sin_info["shape"], dtype=DTYPE_MAP[sin_info["dtype"]]) * (sin_info["range"][1] - sin_info["range"][0]) + sin_info["range"][0]

        input_groups.append([q, k, cos, sin])
    return input_groups


def get_init_inputs():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level3_0_ApplyRotaryPosEmb.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    init_groups = []
    for case in cases:
        entries = case.get("init_inputs", [])
        layout_info = entries[0]
        layout = layout_info["value"]
        init_groups.append([layout])
    return init_groups
