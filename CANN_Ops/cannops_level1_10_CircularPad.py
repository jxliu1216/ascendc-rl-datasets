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

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """CPU 参考：与 aclnn CircularPad2d / CircularPad3d 语义一致（circular 填充最后两维或三维）。"""

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, padding):
        pad = tuple((int(p) for p in padding))
        return F.pad(x, pad, mode='circular')

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level1_10_CircularPad.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        x_info = inputs[0]
        padding_info = inputs[1]

        if "data" in x_info:
            x = torch.tensor(x_info["data"], dtype=DTYPE_MAP[x_info["dtype"]]).reshape(x_info["shape"])
        else:
            _dt = DTYPE_MAP[x_info["dtype"]]
            if _dt in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
                x = torch.randint(x_info["range"][0], x_info["range"][1] + 1, tuple(x_info["shape"]), dtype=_dt)
            elif _dt == torch.bool:
                x = torch.rand(x_info["shape"]) > 0.5
            else:
                x = torch.randn(x_info["shape"], dtype=_dt)
        padding = padding_info["value"]

        input_groups.append([x, padding])
    return input_groups


def get_init_inputs():
    return []
