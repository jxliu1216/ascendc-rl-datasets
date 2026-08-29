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
import torch.nn.functional as F

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input: torch.Tensor, mulscalar1: torch.Tensor, mulscalar2: torch.Tensor, mulscalar3: torch.Tensor) -> torch.Tensor:
        mul1_res = input * mulscalar1
        sigmoid_res = 1 / (1 + torch.exp(-mul1_res))
        mul_2_res = sigmoid_res * mulscalar2
        output = mul_2_res + mulscalar3
        return output

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level2_132_MulSigmoidMulAddCustom.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        input_info = inputs[0]
        mulscalar1_info = inputs[1]
        mulscalar2_info = inputs[2]
        mulscalar3_info = inputs[3]

        if "data" in input_info:
            input = torch.tensor(input_info["data"], dtype=DTYPE_MAP[input_info["dtype"]]).reshape(input_info["shape"])
        else:
            input = torch.randn(input_info["shape"], dtype=DTYPE_MAP[input_info["dtype"]])
        if "data" in mulscalar1_info:
            mulscalar1 = torch.tensor(mulscalar1_info["data"], dtype=DTYPE_MAP[mulscalar1_info["dtype"]]).reshape(mulscalar1_info["shape"])
        else:
            mulscalar1 = torch.rand(mulscalar1_info["shape"], dtype=DTYPE_MAP[mulscalar1_info["dtype"]]) * (mulscalar1_info["range"][1] - mulscalar1_info["range"][0]) + mulscalar1_info["range"][0]
        if "data" in mulscalar2_info:
            mulscalar2 = torch.tensor(mulscalar2_info["data"], dtype=DTYPE_MAP[mulscalar2_info["dtype"]]).reshape(mulscalar2_info["shape"])
        else:
            mulscalar2 = torch.rand(mulscalar2_info["shape"], dtype=DTYPE_MAP[mulscalar2_info["dtype"]]) * (mulscalar2_info["range"][1] - mulscalar2_info["range"][0]) + mulscalar2_info["range"][0]
        if "data" in mulscalar3_info:
            mulscalar3 = torch.tensor(mulscalar3_info["data"], dtype=DTYPE_MAP[mulscalar3_info["dtype"]]).reshape(mulscalar3_info["shape"])
        else:
            mulscalar3 = torch.rand(mulscalar3_info["shape"], dtype=DTYPE_MAP[mulscalar3_info["dtype"]]) * (mulscalar3_info["range"][1] - mulscalar3_info["range"][0]) + mulscalar3_info["range"][0]

        input_groups.append([input, mulscalar1, mulscalar2, mulscalar3])
    return input_groups


def get_init_inputs():
    return []
