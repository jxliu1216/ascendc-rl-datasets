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

from typing import List
import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self, gamma: torch.Tensor, epsilon: float):
        super(Model, self).__init__()
        self.gamma = gamma
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        out = x * torch.rsqrt(x.type(torch.float32).pow(2).mean(-1, keepdim=True) + self.epsilon).type(torch.float32)
        golden = out * self.gamma.type(torch.float32)
        return [golden]

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level2_138_RmsNorm.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        x_info = inputs[0]

        if "data" in x_info:
            x = torch.tensor(x_info["data"], dtype=DTYPE_MAP[x_info["dtype"]]).reshape(x_info["shape"])
        else:
            x = torch.rand(x_info["shape"], dtype=DTYPE_MAP[x_info["dtype"]])

        input_groups.append([x])
    return input_groups


def get_init_inputs():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level2_138_RmsNorm.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    init_groups = []
    for case in cases:
        entries = case.get("init_inputs", [])
        gamma_info = entries[0]
        epsilon_info = entries[1]
        if "data" in gamma_info:
            gamma = torch.tensor(gamma_info["data"], dtype=DTYPE_MAP[gamma_info["dtype"]]).reshape(gamma_info["shape"])
        else:
            gamma = torch.rand(gamma_info["shape"], dtype=DTYPE_MAP[gamma_info["dtype"]])
        epsilon = epsilon_info["value"]
        init_groups.append([gamma, epsilon])
    return init_groups
