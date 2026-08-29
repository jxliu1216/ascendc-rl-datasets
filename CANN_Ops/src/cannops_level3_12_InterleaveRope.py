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
CPU 金标准与 ops-transformer `interleave_rope/README.md` 公式一致：
先对最后一维做 interleave（view 为 [..., D/2, 2] 再交换最后两维），再
y = q * cos + RotateHalf(q) * sin，其中 RotateHalf 为 concat([-q[..., D/2:], q[..., :D/2]], dim=-1)。
在 float32 上计算后 cast 回输入 dtype；cos/sin 按广播规则扩展到 q 的形状。
"""
from typing import List, Tuple
import torch
import torch.nn as nn

def _interleave_last_dim(x: torch.Tensor) -> torch.Tensor:
    *rest, d = x.shape
    assert d % 2 == 0
    return x.reshape(*rest, d // 2, 2).transpose(-1, -2).contiguous().reshape(*rest, d)

def _rotate_half(q: torch.Tensor) -> torch.Tensor:
    d = q.shape[-1]
    half = d // 2
    return torch.cat([-q[..., half:], q[..., :half]], dim=-1)

def golden_interleave_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    od = x.dtype
    q = _interleave_last_dim(x.float())
    cos_b = torch.broadcast_to(cos.float(), q.shape)
    sin_b = torch.broadcast_to(sin.float(), q.shape)
    y = q * cos_b + _rotate_half(q) * sin_b
    return y.to(od)

class Model(nn.Module):

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, _out: torch.Tensor) -> List[torch.Tensor]:
        del _out
        return [golden_interleave_rope(x, cos, sin)]

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level3_12_InterleaveRope.json')
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
        if "data" in _out_info:
            _out = torch.tensor(_out_info["data"], dtype=DTYPE_MAP[_out_info["dtype"]]).reshape(_out_info["shape"])
        else:
            _out = torch.randn(_out_info["shape"], dtype=DTYPE_MAP[_out_info["dtype"]]) * _out_info["std"] + _out_info["mean"]
        if _out_info.get("inject"):
            _f = _out.reshape(-1)
            _f[0] = float(_out_info["inject"])
            _out = _f.reshape(_out.shape)

        input_groups.append([x, cos, sin, _out])
    return input_groups


def get_init_inputs():
    return []
