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

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        return torch.pow(x1.to(torch.float32), x2.to(torch.float32)).to(x1.dtype)

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level1_52_Pows.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        x1_info = inputs[0]
        x2_info = inputs[1]

        if "data" in x1_info:
            x1 = torch.tensor(x1_info["data"], dtype=DTYPE_MAP[x1_info["dtype"]]).reshape(x1_info["shape"])
        else:
            x1 = torch.abs(torch.randn(x1_info["shape"], dtype=DTYPE_MAP[x1_info["dtype"]]) * x1_info["sigma"])
        if "data" in x2_info:
            x2 = torch.tensor(x2_info["data"], dtype=DTYPE_MAP[x2_info["dtype"]]).reshape(x2_info["shape"])
        else:
            x2 = torch.randn(x2_info["shape"], dtype=DTYPE_MAP[x2_info["dtype"]]) * x2_info["std"] + x2_info["mean"]

        input_groups.append([x1, x2])
    return input_groups


def get_init_inputs():
    return []
