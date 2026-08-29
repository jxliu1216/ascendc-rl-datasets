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

from typing import List, Optional
import torch
import torch.nn as nn
import math

class Model(nn.Module):

    def __init__(self, n: int, incx: int, incy: int):
        super(Model, self).__init__()
        self.n = n
        self.incx = incx
        self.incy = incy

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_indices = torch.arange(self.n, device=x.device) * self.incx
        y_indices = torch.arange(self.n, device=x.device) * self.incy
        out_size = y_indices.max().item() + 1
        out = torch.zeros(out_size, dtype=x.dtype, device=x.device)
        x_flat = x.flatten()
        out[y_indices] = x_flat[x_indices]
        return out

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level1_9_Ccopy.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        x_info = inputs[0]

        if "data" in x_info:
            x = torch.tensor(x_info["data"], dtype=DTYPE_MAP[x_info["dtype"]]).reshape(x_info["shape"])
        else:
            x_re = torch.rand(x_info["shape"], dtype=torch.float32)
            x_im = torch.rand(x_info["shape"], dtype=torch.float32)
            x = torch.complex(x_re, x_im).to(DTYPE_MAP[x_info["dtype"]])

        input_groups.append([x])
    return input_groups


def get_init_inputs():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level1_9_Ccopy.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    init_groups = []
    for case in cases:
        entries = case.get("init_inputs", [])
        n_info = entries[0]
        incx_info = entries[1]
        incy_info = entries[2]
        n = n_info["value"]
        incx = incx_info["value"]
        incy = incy_info["value"]
        init_groups.append([n, incx, incy])
    return init_groups
