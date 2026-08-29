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

    def forward(self, add_0_input0: torch.Tensor, add_0_input1: torch.Tensor, mult_0_input1: torch.Tensor, mult_1_input1: torch.Tensor, mult_2_input1: torch.Tensor) -> torch.Tensor:
        add_res = add_0_input0 + add_0_input1
        mul1_res = add_res * mult_0_input1
        sig_res = 1 / (1 + torch.exp(-mul1_res))
        mul2_res = sig_res * mult_1_input1
        mul3_res = mul2_res * mult_2_input1
        output = torch.sum(mul3_res, dim=1)
        return output

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level2_12_AddSigmoidMulReduceSumD.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        add_0_input0_info = inputs[0]
        add_0_input1_info = inputs[1]
        mult_0_input1_info = inputs[2]
        mult_1_input1_info = inputs[3]
        mult_2_input1_info = inputs[4]

        if "data" in add_0_input0_info:
            add_0_input0 = torch.tensor(add_0_input0_info["data"], dtype=DTYPE_MAP[add_0_input0_info["dtype"]]).reshape(add_0_input0_info["shape"])
        else:
            add_0_input0 = torch.randn(add_0_input0_info["shape"], dtype=DTYPE_MAP[add_0_input0_info["dtype"]]) * add_0_input0_info["std"] + add_0_input0_info["mean"]
        if "data" in add_0_input1_info:
            add_0_input1 = torch.tensor(add_0_input1_info["data"], dtype=DTYPE_MAP[add_0_input1_info["dtype"]]).reshape(add_0_input1_info["shape"])
        else:
            add_0_input1 = torch.randn(add_0_input1_info["shape"], dtype=DTYPE_MAP[add_0_input1_info["dtype"]]) * add_0_input1_info["std"] + add_0_input1_info["mean"]
        if "data" in mult_0_input1_info:
            mult_0_input1 = torch.tensor(mult_0_input1_info["data"], dtype=DTYPE_MAP[mult_0_input1_info["dtype"]]).reshape(mult_0_input1_info["shape"])
        else:
            mult_0_input1 = torch.rand(mult_0_input1_info["shape"], dtype=DTYPE_MAP[mult_0_input1_info["dtype"]]) * (mult_0_input1_info["range"][1] - mult_0_input1_info["range"][0]) + mult_0_input1_info["range"][0]
        if "data" in mult_1_input1_info:
            mult_1_input1 = torch.tensor(mult_1_input1_info["data"], dtype=DTYPE_MAP[mult_1_input1_info["dtype"]]).reshape(mult_1_input1_info["shape"])
        else:
            mult_1_input1 = torch.randn(mult_1_input1_info["shape"], dtype=DTYPE_MAP[mult_1_input1_info["dtype"]]) * mult_1_input1_info["std"] + mult_1_input1_info["mean"]
        if "data" in mult_2_input1_info:
            mult_2_input1 = torch.tensor(mult_2_input1_info["data"], dtype=DTYPE_MAP[mult_2_input1_info["dtype"]]).reshape(mult_2_input1_info["shape"])
        else:
            mult_2_input1 = torch.randn(mult_2_input1_info["shape"], dtype=DTYPE_MAP[mult_2_input1_info["dtype"]]) * mult_2_input1_info["std"] + mult_2_input1_info["mean"]

        input_groups.append([add_0_input0, add_0_input1, mult_0_input1, mult_1_input1, mult_2_input1])
    return input_groups


def get_init_inputs():
    return []
