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
    """
    实现Fill算子功能的模型。
    """

    def __init__(self):
        """
        初始化模型。
        """
        super(Model, self).__init__()

    def forward(self, dims: List[int], value: torch.Tensor) -> torch.Tensor:
        """
        实现Fill算子功能。

        Args:
            dims: 用于指定输出张量的形状
            value: 用于填充张量的标量值

        Returns:
            用指定标量值填充的张量
        """
        return torch.full(dims, value.item(), dtype=value.dtype)

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level1_21_Fill.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        dims_info = inputs[0]
        value_info = inputs[1]

        dims = dims_info["value"]
        if "data" in value_info:
            value = torch.tensor(value_info["data"], dtype=DTYPE_MAP[value_info["dtype"]]).reshape(value_info["shape"])
        else:
            _dt = DTYPE_MAP[value_info["dtype"]]
            if _dt in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
                value = torch.randint(value_info["range"][0], value_info["range"][1] + 1, tuple(value_info["shape"]), dtype=_dt)
            elif _dt == torch.bool:
                value = torch.rand(value_info["shape"]) > 0.5
            else:
                value = torch.rand(value_info["shape"], dtype=_dt) * (value_info["range"][1] - value_info["range"][0]) + value_info["range"][0]

        input_groups.append([dims, value])
    return input_groups


def get_init_inputs():
    return []
