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

    def forward(self, x: torch.Tensor, y: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, epsilon: float) -> torch.Tensor:
        add = x + y
        x_shape = x.shape
        normalized_shape = (x_shape[-1],)
        output = F.layer_norm(add, normalized_shape, gamma, beta, epsilon)
        return output

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level2_133_PreLayerNorm.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        x_info = inputs[0]
        y_info = inputs[1]
        gamma_info = inputs[2]
        beta_info = inputs[3]
        epsilon_info = inputs[4]

        if "data" in x_info:
            x = torch.tensor(x_info["data"], dtype=DTYPE_MAP[x_info["dtype"]]).reshape(x_info["shape"])
        else:
            x = torch.randn(x_info["shape"], dtype=DTYPE_MAP[x_info["dtype"]])
        if "data" in y_info:
            y = torch.tensor(y_info["data"], dtype=DTYPE_MAP[y_info["dtype"]]).reshape(y_info["shape"])
        else:
            y = torch.randn(y_info["shape"], dtype=DTYPE_MAP[y_info["dtype"]])
        if "data" in gamma_info:
            gamma = torch.tensor(gamma_info["data"], dtype=DTYPE_MAP[gamma_info["dtype"]]).reshape(gamma_info["shape"])
        else:
            gamma = torch.randn(gamma_info["shape"], dtype=DTYPE_MAP[gamma_info["dtype"]])
        if "data" in beta_info:
            beta = torch.tensor(beta_info["data"], dtype=DTYPE_MAP[beta_info["dtype"]]).reshape(beta_info["shape"])
        else:
            beta = torch.randn(beta_info["shape"], dtype=DTYPE_MAP[beta_info["dtype"]])
        epsilon = torch.rand(1).item() * (epsilon_info["range"][1] - epsilon_info["range"][0]) + epsilon_info["range"][0]

        input_groups.append([x, y, gamma, beta, epsilon])
    return input_groups


def get_init_inputs():
    return []
