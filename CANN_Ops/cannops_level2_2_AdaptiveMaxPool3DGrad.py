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

    def forward(self, y_grad: torch.Tensor, x: torch.Tensor, argmax: torch.Tensor) -> torch.Tensor:
        res = torch.ops.aten.adaptive_max_pool3d_backward(y_grad.to(torch.float32), x.to(torch.float32), argmax.to(torch.int64))
        return res.to(x.dtype)

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level2_2_AdaptiveMaxPool3DGrad.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        y_grad_info = inputs[0]
        x_info = inputs[1]
        argmax_info = inputs[2]

        if "data" in y_grad_info:
            y_grad = torch.tensor(y_grad_info["data"], dtype=DTYPE_MAP[y_grad_info["dtype"]]).reshape(y_grad_info["shape"])
        else:
            y_grad = torch.randn(y_grad_info["shape"], dtype=DTYPE_MAP[y_grad_info["dtype"]]) * y_grad_info["std"] + y_grad_info["mean"]
        if "data" in x_info:
            x = torch.tensor(x_info["data"], dtype=DTYPE_MAP[x_info["dtype"]]).reshape(x_info["shape"])
        else:
            x = torch.randn(x_info["shape"], dtype=DTYPE_MAP[x_info["dtype"]])
        if "data" in argmax_info:
            argmax = torch.tensor(argmax_info["data"], dtype=DTYPE_MAP[argmax_info["dtype"]]).reshape(argmax_info["shape"])
        else:
            argmax = torch.randint(argmax_info["range"][0], argmax_info["range"][1] + 1, tuple(argmax_info["shape"]), dtype=DTYPE_MAP[argmax_info["dtype"]])

        input_groups.append([y_grad, x, argmax])
    return input_groups


def get_init_inputs():
    return []
