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

    def forward(self, dy: torch.Tensor, input: torch.Tensor) -> torch.Tensor:
        attr = 1.702
        attr_opp = -attr
        attr_half = attr / 2
        abs_x = torch.abs(input)
        mul_abs_x = abs_x * attr_opp
        exp_x = torch.exp(mul_abs_x)
        add_2 = input * exp_x * attr
        exp_pn_x = torch.exp((input - abs_x) * attr)
        div_up = exp_x + add_2 + exp_pn_x
        div_down = (exp_x + 1.0) ** 2
        res = div_up / div_down
        output = dy * res
        return output

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level2_39_FastGeluGrad.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        dy_info = inputs[0]
        input_info = inputs[1]

        if "data" in dy_info:
            dy = torch.tensor(dy_info["data"], dtype=DTYPE_MAP[dy_info["dtype"]]).reshape(dy_info["shape"])
        else:
            dy = torch.rand(dy_info["shape"], dtype=DTYPE_MAP[dy_info["dtype"]])
        if "data" in input_info:
            input = torch.tensor(input_info["data"], dtype=DTYPE_MAP[input_info["dtype"]]).reshape(input_info["shape"])
        else:
            input = torch.rand(input_info["shape"], dtype=DTYPE_MAP[input_info["dtype"]])

        input_groups.append([dy, input])
    return input_groups


def get_init_inputs():
    return []
