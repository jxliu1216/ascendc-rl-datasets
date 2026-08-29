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

    def forward(self, input1: torch.Tensor, scale: float) -> torch.Tensor:
        return input1 * torch.sigmoid(input1 * scale)

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level2_147_Swish.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        input1_info = inputs[0]
        scale_info = inputs[1]

        if "data" in input1_info:
            input1 = torch.tensor(input1_info["data"], dtype=DTYPE_MAP[input1_info["dtype"]]).reshape(input1_info["shape"])
        else:
            input1 = torch.randn(input1_info["shape"], dtype=DTYPE_MAP[input1_info["dtype"]])
        scale = scale_info["value"]

        input_groups.append([input1, scale])
    return input_groups


def get_init_inputs():
    return []
