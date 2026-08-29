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
    """使用 PyTorch 原生算子的参考实现（golden model）。"""

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, grad, x, grid, interpolation_mode='bilinear', padding_mode='zeros', align_corners=False):
        input_dtype = x.dtype
        x_float = x.detach().float().requires_grad_(True)
        grid_float = grid.detach().float().requires_grad_(True)
        grad_float = grad.float()
        output = torch.nn.functional.grid_sample(x_float, grid_float, mode=interpolation_mode, padding_mode=padding_mode, align_corners=align_corners)
        output.backward(grad_float)
        dx = x_float.grad.to(input_dtype)
        dgrid = grid_float.grad.to(input_dtype)
        return (dx, dgrid)

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level2_107_GridSampler2DGrad.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        grad_info = inputs[0]
        x_info = inputs[1]
        grid_info = inputs[2]
        interpolation_mode_info = inputs[3]
        padding_mode_info = inputs[4]
        align_corners_info = inputs[5]

        if "data" in grad_info:
            grad = torch.tensor(grad_info["data"], dtype=DTYPE_MAP[grad_info["dtype"]]).reshape(grad_info["shape"])
        else:
            grad = torch.randn(grad_info["shape"], dtype=DTYPE_MAP[grad_info["dtype"]])
        if "data" in x_info:
            x = torch.tensor(x_info["data"], dtype=DTYPE_MAP[x_info["dtype"]]).reshape(x_info["shape"])
        else:
            x = torch.randn(x_info["shape"], dtype=DTYPE_MAP[x_info["dtype"]])
        if "data" in grid_info:
            grid = torch.tensor(grid_info["data"], dtype=DTYPE_MAP[grid_info["dtype"]]).reshape(grid_info["shape"])
        else:
            grid = torch.rand(grid_info["shape"], dtype=DTYPE_MAP[grid_info["dtype"]]) * (grid_info["range"][1] - grid_info["range"][0]) + grid_info["range"][0]
        interpolation_mode = interpolation_mode_info["value"]
        padding_mode = padding_mode_info["value"]
        align_corners = align_corners_info["value"]

        input_groups.append([grad, x, grid, interpolation_mode, padding_mode, align_corners])
    return input_groups


def get_init_inputs():
    return []
