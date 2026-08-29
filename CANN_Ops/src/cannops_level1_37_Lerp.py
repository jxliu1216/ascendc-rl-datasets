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

    def forward(self, start: torch.Tensor, end: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        output = torch.lerp(start, end, weight)
        return output

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level1_37_Lerp.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        start_info = inputs[0]
        end_info = inputs[1]
        weight_info = inputs[2]

        if "data" in start_info:
            start = torch.tensor(start_info["data"], dtype=DTYPE_MAP[start_info["dtype"]]).reshape(start_info["shape"])
        else:
            start = torch.rand(start_info["shape"], dtype=DTYPE_MAP[start_info["dtype"]])
        if "data" in end_info:
            end = torch.tensor(end_info["data"], dtype=DTYPE_MAP[end_info["dtype"]]).reshape(end_info["shape"])
        else:
            end = torch.rand(end_info["shape"], dtype=DTYPE_MAP[end_info["dtype"]])
        if "data" in weight_info:
            weight = torch.tensor(weight_info["data"], dtype=DTYPE_MAP[weight_info["dtype"]]).reshape(weight_info["shape"])
        else:
            weight = torch.rand(weight_info["shape"], dtype=DTYPE_MAP[weight_info["dtype"]])

        input_groups.append([start, end, weight])
    return input_groups


def get_init_inputs():
    return []
