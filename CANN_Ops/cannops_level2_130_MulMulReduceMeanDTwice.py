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

    def forward(self, mul0_input0: torch.Tensor, mul0_input1: torch.Tensor, mul1_input0: torch.Tensor, add_y: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        mul_res = mul0_input0 * mul0_input1 * mul1_input0
        reduce_mean_0 = torch.mean(mul_res, dim=1, keepdim=True)
        diff = mul_res - reduce_mean_0
        muld_res = diff * diff
        x2 = torch.mean(muld_res, dim=1, keepdim=True)
        reduce_mean_1 = gamma / torch.sqrt(x2 + add_y)
        output = beta - reduce_mean_1 * reduce_mean_0 + reduce_mean_1 * mul_res
        return output

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level2_130_MulMulReduceMeanDTwice.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        mul0_input0_info = inputs[0]
        mul0_input1_info = inputs[1]
        mul1_input0_info = inputs[2]
        add_y_info = inputs[3]
        gamma_info = inputs[4]
        beta_info = inputs[5]

        if "data" in mul0_input0_info:
            mul0_input0 = torch.tensor(mul0_input0_info["data"], dtype=DTYPE_MAP[mul0_input0_info["dtype"]]).reshape(mul0_input0_info["shape"])
        else:
            mul0_input0 = torch.rand(mul0_input0_info["shape"], dtype=DTYPE_MAP[mul0_input0_info["dtype"]])
        if "data" in mul0_input1_info:
            mul0_input1 = torch.tensor(mul0_input1_info["data"], dtype=DTYPE_MAP[mul0_input1_info["dtype"]]).reshape(mul0_input1_info["shape"])
        else:
            mul0_input1 = torch.rand(mul0_input1_info["shape"], dtype=DTYPE_MAP[mul0_input1_info["dtype"]])
        if "data" in mul1_input0_info:
            mul1_input0 = torch.tensor(mul1_input0_info["data"], dtype=DTYPE_MAP[mul1_input0_info["dtype"]]).reshape(mul1_input0_info["shape"])
        else:
            mul1_input0 = torch.full(mul1_input0_info["shape"], mul1_input0_info["fill"], dtype=DTYPE_MAP[mul1_input0_info["dtype"]])
        if "data" in add_y_info:
            add_y = torch.tensor(add_y_info["data"], dtype=DTYPE_MAP[add_y_info["dtype"]]).reshape(add_y_info["shape"])
        else:
            add_y = torch.full(add_y_info["shape"], add_y_info["fill"], dtype=DTYPE_MAP[add_y_info["dtype"]])
        if "data" in gamma_info:
            gamma = torch.tensor(gamma_info["data"], dtype=DTYPE_MAP[gamma_info["dtype"]]).reshape(gamma_info["shape"])
        else:
            gamma = torch.rand(gamma_info["shape"], dtype=DTYPE_MAP[gamma_info["dtype"]])
        if "data" in beta_info:
            beta = torch.tensor(beta_info["data"], dtype=DTYPE_MAP[beta_info["dtype"]]).reshape(beta_info["shape"])
        else:
            beta = torch.rand(beta_info["shape"], dtype=DTYPE_MAP[beta_info["dtype"]])

        input_groups.append([mul0_input0, mul0_input1, mul1_input0, add_y, gamma, beta])
    return input_groups


def get_init_inputs():
    return []
