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

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
        output = torch.eq(input1, input2)
        output = output.to(dtype=torch.int)
        return output

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level1_19_Equal.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        input1_info = inputs[0]
        input2_info = inputs[1]

        if "data" in input1_info:
            input1 = torch.tensor(input1_info["data"], dtype=DTYPE_MAP[input1_info["dtype"]]).reshape(input1_info["shape"])
        else:
            _dt = DTYPE_MAP[input1_info["dtype"]]
            if _dt in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
                input1 = torch.randint(input1_info["range"][0], input1_info["range"][1] + 1, tuple(input1_info["shape"]), dtype=_dt)
            elif _dt == torch.bool:
                input1 = torch.rand(input1_info["shape"]) > 0.5
            else:
                input1 = torch.randn(input1_info["shape"], dtype=_dt)
        if "data" in input2_info:
            input2 = torch.tensor(input2_info["data"], dtype=DTYPE_MAP[input2_info["dtype"]]).reshape(input2_info["shape"])
        else:
            _dt = DTYPE_MAP[input2_info["dtype"]]
            if _dt in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
                input2 = torch.randint(input2_info["range"][0], input2_info["range"][1] + 1, tuple(input2_info["shape"]), dtype=_dt)
            elif _dt == torch.bool:
                input2 = torch.rand(input2_info["shape"]) > 0.5
            else:
                input2 = torch.randn(input2_info["shape"], dtype=_dt)

        input_groups.append([input1, input2])
    return input_groups


def get_init_inputs():
    return []
