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
    """
    实现GatherV3算子功能的模型。
    """

    def __init__(self):
        """
        初始化模型。
        """
        super(Model, self).__init__()

    def forward(self, self_tensor: torch.Tensor, axis: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """
        实现GatherV3算子功能。

        Args:
            self_tensor: 第一个输入张量
            indices: 索引张量
            axis: 轴张量

        Returns:
            GatherV3算子的输出
        """
        output = torch.index_select(self_tensor, dim=axis, index=indices)
        return output

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level1_24_GatherV3.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        self_tensor_info = inputs[0]
        axis_info = inputs[1]
        indices_info = inputs[2]

        if "data" in self_tensor_info:
            self_tensor = torch.tensor(self_tensor_info["data"], dtype=DTYPE_MAP[self_tensor_info["dtype"]]).reshape(self_tensor_info["shape"])
        else:
            self_tensor = torch.randn(self_tensor_info["shape"], dtype=DTYPE_MAP[self_tensor_info["dtype"]])
        if "data" in axis_info:
            axis = torch.tensor(axis_info["data"], dtype=DTYPE_MAP[axis_info["dtype"]]).reshape(axis_info["shape"])
        else:
            axis = torch.randint(axis_info["range"][0], axis_info["range"][1] + 1, tuple(axis_info["shape"]), dtype=DTYPE_MAP[axis_info["dtype"]])
        if "data" in indices_info:
            indices = torch.tensor(indices_info["data"], dtype=DTYPE_MAP[indices_info["dtype"]]).reshape(indices_info["shape"])
        else:
            indices = torch.randint(indices_info["range"][0], indices_info["range"][1] + 1, tuple(indices_info["shape"]), dtype=DTYPE_MAP[indices_info["dtype"]])

        input_groups.append([self_tensor, axis, indices])
    return input_groups


def get_init_inputs():
    return []
