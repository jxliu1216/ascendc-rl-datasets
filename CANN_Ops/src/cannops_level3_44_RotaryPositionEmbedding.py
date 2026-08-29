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
CPU 金标准与 ops-transformer `posembedding/rotary_position_embedding/README.md` 中四种 mode 公式一致：
0 half，1 interleave，2 quarter（D 能被 4 整除），3 interleave-half（x_part1*cos + x_part2*sin）。
在 float32 上计算再 cast 回输入 dtype；cos/sin 广播到与 x 同形。
"""
from typing import List
import torch
import torch.nn as nn

def golden_rotary_position_embedding(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, mode: int) -> torch.Tensor:
    od = x.dtype
    xf = x.float()
    cos_b = torch.broadcast_to(cos.float(), xf.shape)
    sin_b = torch.broadcast_to(sin.float(), xf.shape)
    d = xf.shape[-1]
    if mode == 0:
        half = d // 2
        x1 = xf[..., :half]
        x2 = xf[..., half:]
        x_rot = torch.cat((-x2, x1), dim=-1)
        y = xf * cos_b + x_rot * sin_b
    elif mode == 1:
        x1 = xf[..., ::2]
        x2 = xf[..., 1::2]
        x_rot = torch.stack((-x2, x1), dim=-1).reshape(*xf.shape)
        y = xf * cos_b + x_rot * sin_b
    elif mode == 2:
        q = d // 4
        x1 = xf[..., :q]
        x2 = xf[..., q:2 * q]
        x3 = xf[..., 2 * q:3 * q]
        x4 = xf[..., 3 * q:]
        x_rot = torch.cat((-x2, x1, -x4, x3), dim=-1)
        y = xf * cos_b + x_rot * sin_b
    elif mode == 3:
        x1 = xf[..., ::2]
        x2 = xf[..., 1::2]
        x_part1 = torch.cat((x1, x2), dim=-1)
        x_part2 = torch.cat((-x2, x1), dim=-1)
        y = x_part1 * cos_b + x_part2 * sin_b
    else:
        raise ValueError(f'unsupported mode {mode}')
    return y.to(od)

class Model(nn.Module):

    def __init__(self, mode: int=0):
        super().__init__()
        self.mode = int(mode)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, _out: torch.Tensor) -> List[torch.Tensor]:
        del _out
        return [golden_rotary_position_embedding(x, cos, sin, self.mode)]

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level3_44_RotaryPositionEmbedding.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        x_info = inputs[0]
        cos_info = inputs[1]
        sin_info = inputs[2]
        _out_info = inputs[3]

        if "data" in x_info:
            x = torch.tensor(x_info["data"], dtype=DTYPE_MAP[x_info["dtype"]]).reshape(x_info["shape"])
        else:
            x = torch.rand(x_info["shape"], dtype=DTYPE_MAP[x_info["dtype"]]) * (x_info["range"][1] - x_info["range"][0]) + x_info["range"][0]
        if "data" in cos_info:
            cos = torch.tensor(cos_info["data"], dtype=DTYPE_MAP[cos_info["dtype"]]).reshape(cos_info["shape"])
        else:
            cos = torch.rand(cos_info["shape"], dtype=DTYPE_MAP[cos_info["dtype"]]) * (cos_info["range"][1] - cos_info["range"][0]) + cos_info["range"][0]
        if "data" in sin_info:
            sin = torch.tensor(sin_info["data"], dtype=DTYPE_MAP[sin_info["dtype"]]).reshape(sin_info["shape"])
        else:
            sin = torch.rand(sin_info["shape"], dtype=DTYPE_MAP[sin_info["dtype"]]) * (sin_info["range"][1] - sin_info["range"][0]) + sin_info["range"][0]
        _out = torch.empty(_out_info["shape"], dtype=DTYPE_MAP[_out_info["dtype"]])

        input_groups.append([x, cos, sin, _out])
    return input_groups


def get_init_inputs():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level3_44_RotaryPositionEmbedding.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    init_groups = []
    for case in cases:
        entries = case.get("init_inputs", [])
        mode_info = entries[0]
        mode = mode_info["value"]
        init_groups.append([mode])
    return init_groups
