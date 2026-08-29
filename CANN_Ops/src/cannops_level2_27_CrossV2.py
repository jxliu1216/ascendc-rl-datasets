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

class Model(nn.Module):
    """
    实现CrossV2功能的golden模型。
    """

    def __init__(self):
        """
        初始化模型。
        """
        super(Model, self).__init__()

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, dim: int):
        """
        实现CrossV2功能。

        Args:
            x1 (Tensor): 第一个输入张量，形状为 [B, D1, D2, ..., Dk]，其中最后一个维度必须 ≥ 3。
            x2 (Tensor): 第二个输入张量，形状与 x1 完全一致，用于与 x1 进行叉积运算。
            dim (int): 指定进行叉积运算的维度索引。支持负索引（如 -1 表示倒数第一维）。必须为整数。
                   - 叉积在该维度的最后三个元素上执行。
                   - 若 dim 为 -1，则在倒数第一维进行叉积。
                   - 若维度大小 < 3，将引发错误。

        Returns:
            y (Tensor): 输出张量，形状与输入张量 x1 相同，表示在指定维度上的叉积结果。
        """
        compute_dtype = x1.dtype
        if compute_dtype in (torch.float16, torch.bfloat16):
            x1 = x1.to(torch.float)
            x2 = x2.to(torch.float)
        y = torch.cross(x1, x2, dim=dim)
        if compute_dtype in (torch.float16, torch.bfloat16):
            y = y.to(compute_dtype)
        return y

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level2_27_CrossV2.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        x1_info = inputs[0]
        x2_info = inputs[1]
        dim_info = inputs[2]

        if "data" in x1_info:
            x1 = torch.tensor(x1_info["data"], dtype=DTYPE_MAP[x1_info["dtype"]]).reshape(x1_info["shape"])
        else:
            _dt = DTYPE_MAP[x1_info["dtype"]]
            if _dt in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
                x1 = torch.randint(x1_info["range"][0], x1_info["range"][1] + 1, tuple(x1_info["shape"]), dtype=_dt)
            elif _dt == torch.bool:
                x1 = torch.rand(x1_info["shape"]) > 0.5
            else:
                x1 = torch.rand(x1_info["shape"], dtype=_dt) * (x1_info["range"][1] - x1_info["range"][0]) + x1_info["range"][0]
        if "data" in x2_info:
            x2 = torch.tensor(x2_info["data"], dtype=DTYPE_MAP[x2_info["dtype"]]).reshape(x2_info["shape"])
        else:
            _dt = DTYPE_MAP[x2_info["dtype"]]
            if _dt in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
                x2 = torch.randint(x2_info["range"][0], x2_info["range"][1] + 1, tuple(x2_info["shape"]), dtype=_dt)
            elif _dt == torch.bool:
                x2 = torch.rand(x2_info["shape"]) > 0.5
            else:
                x2 = torch.rand(x2_info["shape"], dtype=_dt) * (x2_info["range"][1] - x2_info["range"][0]) + x2_info["range"][0]
        dim = dim_info["value"]

        input_groups.append([x1, x2, dim])
    return input_groups


def get_init_inputs():
    return []
