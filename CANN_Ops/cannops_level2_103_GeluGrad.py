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
    实现GELU梯度算子功能的模型。
    """

    def __init__(self):
        """
        初始化模型。
        """
        super(Model, self).__init__()

    def forward(self, dy: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        实现GELU梯度算子功能。

        Args:
            dy: 梯度输入张量
            x: 原始输入张量

        Returns:
            GELU梯度计算结果
        """
        orig_dtype = x.dtype
        x = x.float()
        dy = dy.float()
        c0 = -0.07135481627260025
        c1 = -1.5957691216057308
        c2 = 0.2140644488178007
        c3 = 1.5957691216057308
        x_square = x * x
        px_arg = (x_square * c0 + c1) * x
        px = torch.exp(px_arg)
        res0 = (x_square * c2 + c3) * x
        t_denominator = px + 1.0
        t = 1.0 / t_denominator
        resp_intermediate = px * res0 * t.pow(2) + t
        resp = torch.nan_to_num(resp_intermediate, nan=0.0)
        z = dy * resp
        return z.to(orig_dtype)

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level2_103_GeluGrad.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        dy_info = inputs[0]
        x_info = inputs[1]

        if "data" in dy_info:
            dy = torch.tensor(dy_info["data"], dtype=DTYPE_MAP[dy_info["dtype"]]).reshape(dy_info["shape"])
        else:
            dy = torch.rand(dy_info["shape"], dtype=DTYPE_MAP[dy_info["dtype"]])
        if "data" in x_info:
            x = torch.tensor(x_info["data"], dtype=DTYPE_MAP[x_info["dtype"]]).reshape(x_info["shape"])
        else:
            x = torch.rand(x_info["shape"], dtype=DTYPE_MAP[x_info["dtype"]])

        input_groups.append([dy, x])
    return input_groups


def get_init_inputs():
    return []
