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
import torch.nn.functional as F

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, y_grad: torch.Tensor, x: torch.Tensor, dim: int=-1) -> torch.Tensor:
        x = x.to(torch.float32)
        y_grad = y_grad.to(torch.float32)
        x1, x2 = x.chunk(2, dim=dim)
        sigmoid_x1 = torch.sigmoid(x1)
        silu_x1 = x1 * sigmoid_x1
        silu_prime = sigmoid_x1 * (1 + x1 * (1 - sigmoid_x1))
        grad_x1 = y_grad * x2 * silu_prime
        grad_x2 = y_grad * silu_x1
        out = torch.cat([grad_x1, grad_x2], dim=dim)
        out = out.to(x.dtype)
        return out

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level2_145_SwiGluGrad.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        y_grad_info = inputs[0]
        x_info = inputs[1]
        dim_info = inputs[2]

        if "data" in y_grad_info:
            y_grad = torch.tensor(y_grad_info["data"], dtype=DTYPE_MAP[y_grad_info["dtype"]]).reshape(y_grad_info["shape"])
        else:
            y_grad = torch.rand(y_grad_info["shape"], dtype=DTYPE_MAP[y_grad_info["dtype"]])
        if "data" in x_info:
            x = torch.tensor(x_info["data"], dtype=DTYPE_MAP[x_info["dtype"]]).reshape(x_info["shape"])
        else:
            x = torch.rand(x_info["shape"], dtype=DTYPE_MAP[x_info["dtype"]])
        dim = dim_info["value"]

        input_groups.append([y_grad, x, dim])
    return input_groups


def get_init_inputs():
    return []
