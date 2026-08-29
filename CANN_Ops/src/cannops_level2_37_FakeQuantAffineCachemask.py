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
CPU golden：torch.fake_quantize_per_channel_affine（axis=0，与 tiling 中 headNum=dim0 一致）。
mask：round(x/scale)+zero_point（float 路径）落在 [quant_min, quant_max] 内（与核心里比较逻辑近似）。
"""
from typing import List, Tuple
import torch
import torch.nn as nn

def _mask_in_quant_range(x: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor, axis: int, qmin: int, qmax: int) -> torch.Tensor:
    """按通道广播后，判断 round(x/s + zp) 是否在整型量化网格范围内。"""
    if axis != 0:
        raise ValueError('golden 仅对齐本算子 tiling：headNum = x.shape[0]，axis 须为 0')
    c = x.shape[0]
    sc = scale.reshape(c, *[1] * (x.ndim - 1)).float()
    zp = zero_point.reshape(c, *[1] * (x.ndim - 1)).float()
    t = torch.round(x.float() / sc + zp)
    return (t >= qmin) & (t <= qmax)

def golden(x: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor, axis: int, quant_min: int, quant_max: int) -> Tuple[torch.Tensor, torch.Tensor]:
    zp = zero_point.to(torch.int32)
    y = torch.fake_quantize_per_channel_affine(x, scale.float(), zp, axis, quant_min, quant_max)
    m = _mask_in_quant_range(x, scale, zero_point, axis, quant_min, quant_max)
    return (y, m)

class Model(nn.Module):

    def __init__(self, axis: int, quant_min: int, quant_max: int):
        super().__init__()
        self.axis = int(axis)
        self.quant_min = int(quant_min)
        self.quant_max = int(quant_max)

    def forward(self, x: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor):
        y, m = golden(x, scale, zero_point, self.axis, self.quant_min, self.quant_max)
        return [y, m]

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level2_37_FakeQuantAffineCachemask.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        x_info = inputs[0]
        scale_info = inputs[1]
        zero_point_info = inputs[2]

        if "data" in x_info:
            x = torch.tensor(x_info["data"], dtype=DTYPE_MAP[x_info["dtype"]]).reshape(x_info["shape"])
        else:
            x = torch.rand(x_info["shape"], dtype=DTYPE_MAP[x_info["dtype"]]) * (x_info["range"][1] - x_info["range"][0]) + x_info["range"][0]
        if "data" in scale_info:
            scale = torch.tensor(scale_info["data"], dtype=DTYPE_MAP[scale_info["dtype"]]).reshape(scale_info["shape"])
        else:
            scale = torch.rand(scale_info["shape"], dtype=DTYPE_MAP[scale_info["dtype"]])
        if "data" in zero_point_info:
            zero_point = torch.tensor(zero_point_info["data"], dtype=DTYPE_MAP[zero_point_info["dtype"]]).reshape(zero_point_info["shape"])
        else:
            zero_point = torch.randint(zero_point_info["range"][0], zero_point_info["range"][1] + 1, tuple(zero_point_info["shape"]), dtype=DTYPE_MAP[zero_point_info["dtype"]])

        input_groups.append([x, scale, zero_point])
    return input_groups


def get_init_inputs():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level2_37_FakeQuantAffineCachemask.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    init_groups = []
    for case in cases:
        entries = case.get("init_inputs", [])
        axis_info = entries[0]
        quant_min_info = entries[1]
        quant_max_info = entries[2]
        axis = axis_info["value"]
        quant_min = quant_min_info["value"]
        quant_max = quant_max_info["value"]
        init_groups.append([axis, quant_min, quant_max])
    return init_groups
