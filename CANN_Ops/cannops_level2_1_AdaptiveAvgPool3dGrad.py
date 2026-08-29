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

    def forward(self, y_grad: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        res = torch.ops.aten._adaptive_avg_pool3d_backward(y_grad.to(torch.float32), x.to(torch.float32))
        return res.to(y_grad.dtype)

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level2_1_AdaptiveAvgPool3dGrad.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        y_grad_info = inputs[0]
        x_info = inputs[1]

        if "data" in y_grad_info:
            y_grad = torch.tensor(y_grad_info["data"], dtype=DTYPE_MAP[y_grad_info["dtype"]]).reshape(y_grad_info["shape"])
        else:
            y_grad = torch.randn(y_grad_info["shape"], dtype=DTYPE_MAP[y_grad_info["dtype"]])
        if "data" in x_info:
            x = torch.tensor(x_info["data"], dtype=DTYPE_MAP[x_info["dtype"]]).reshape(x_info["shape"])
        else:
            x = torch.randn(x_info["shape"], dtype=DTYPE_MAP[x_info["dtype"]])

        input_groups.append([y_grad, x])
    return input_groups


def get_init_inputs():
    return []
