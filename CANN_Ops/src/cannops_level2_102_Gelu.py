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
    实现GELU算子功能的模型。
    """

    def __init__(self):
        """
        初始化模型。
        """
        super(Model, self).__init__()

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        实现GELU算子功能。

        Args:
            input_tensor: 输入张量

        Returns:
            应用GELU激活函数后的张量
        """
        return torch.nn.functional.gelu(input_tensor)

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level2_102_Gelu.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        input_tensor_info = inputs[0]

        if "data" in input_tensor_info:
            input_tensor = torch.tensor(input_tensor_info["data"], dtype=DTYPE_MAP[input_tensor_info["dtype"]]).reshape(input_tensor_info["shape"])
        else:
            input_tensor = torch.randn(input_tensor_info["shape"], dtype=DTYPE_MAP[input_tensor_info["dtype"]])

        input_groups.append([input_tensor])
    return input_groups


def get_init_inputs():
    return []
