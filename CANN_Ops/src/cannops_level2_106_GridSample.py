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

    def forward(self, x: torch.Tensor, grid: torch.Tensor, mode: str='bilinear', padding_mode: str='zeros', align_corners: bool=False) -> torch.Tensor:
        input_dtype = x.dtype
        x_float = x.float()
        grid_float = grid.float()
        result = torch.nn.functional.grid_sample(x_float, grid_float, mode=mode, padding_mode=padding_mode, align_corners=align_corners)
        return result.to(input_dtype)

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level2_106_GridSample.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        x_info = inputs[0]
        grid_info = inputs[1]
        mode_info = inputs[2]
        padding_mode_info = inputs[3]
        align_corners_info = inputs[4]

        if "data" in x_info:
            x = torch.tensor(x_info["data"], dtype=DTYPE_MAP[x_info["dtype"]]).reshape(x_info["shape"])
        else:
            x = torch.randn(x_info["shape"], dtype=DTYPE_MAP[x_info["dtype"]])
        if "data" in grid_info:
            grid = torch.tensor(grid_info["data"], dtype=DTYPE_MAP[grid_info["dtype"]]).reshape(grid_info["shape"])
        else:
            grid = torch.rand(grid_info["shape"], dtype=DTYPE_MAP[grid_info["dtype"]]) * (grid_info["range"][1] - grid_info["range"][0]) + grid_info["range"][0]
        mode = mode_info["value"]
        padding_mode = padding_mode_info["value"]
        align_corners = align_corners_info["value"]

        input_groups.append([x, grid, mode, padding_mode, align_corners])
    return input_groups


def get_init_inputs():
    return []
