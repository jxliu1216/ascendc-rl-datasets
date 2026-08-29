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

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = input.to(torch.float16)
        mid_col = input.shape[-1] // 2
        output = input.clone()
        output[:, :, :, mid_col:] = -input[:, :, :, mid_col:]
        return output

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level1_70_StridesliceNegConcatV2.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        input_info = inputs[0]

        if "data" in input_info:
            input = torch.tensor(input_info["data"], dtype=DTYPE_MAP[input_info["dtype"]]).reshape(input_info["shape"])
        else:
            input = torch.randn(input_info["shape"], dtype=DTYPE_MAP[input_info["dtype"]])

        input_groups.append([input])
    return input_groups


def get_init_inputs():
    return []
