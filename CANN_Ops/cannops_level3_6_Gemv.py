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
    实现add算子功能的模型。
    """

    def __init__(self):
        """
        初始化模型。
        """
        super(Model, self).__init__()

    def forward(self, a: torch.Tensor, x: torch.Tensor, y: torch.Tensor, alpha: torch.float32, beta: torch.float32) -> torch.Tensor:
        """
        实现add算子功能。

        Args:
            a: 第一个输入张量
            b: 第二个输入张量

        Returns:
            两个输入张量的和
        """
        output = alpha * (a @ x) + beta * y
        return output

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level3_6_Gemv.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        a_info = inputs[0]
        x_info = inputs[1]
        y_info = inputs[2]
        alpha_info = inputs[3]
        beta_info = inputs[4]

        if "data" in a_info:
            a = torch.tensor(a_info["data"], dtype=DTYPE_MAP[a_info["dtype"]]).reshape(a_info["shape"])
        else:
            a = torch.rand(a_info["shape"], dtype=DTYPE_MAP[a_info["dtype"]]) * (a_info["range"][1] - a_info["range"][0]) + a_info["range"][0]
        if "data" in x_info:
            x = torch.tensor(x_info["data"], dtype=DTYPE_MAP[x_info["dtype"]]).reshape(x_info["shape"])
        else:
            x = torch.rand(x_info["shape"], dtype=DTYPE_MAP[x_info["dtype"]]) * (x_info["range"][1] - x_info["range"][0]) + x_info["range"][0]
        if "data" in y_info:
            y = torch.tensor(y_info["data"], dtype=DTYPE_MAP[y_info["dtype"]]).reshape(y_info["shape"])
        else:
            y = torch.rand(y_info["shape"], dtype=DTYPE_MAP[y_info["dtype"]]) * (y_info["range"][1] - y_info["range"][0]) + y_info["range"][0]
        if "data" in alpha_info:
            alpha = torch.tensor(alpha_info["data"], dtype=DTYPE_MAP[alpha_info["dtype"]]).reshape(alpha_info["shape"])
        else:
            alpha = torch.rand(alpha_info["shape"], dtype=DTYPE_MAP[alpha_info["dtype"]]) * (alpha_info["range"][1] - alpha_info["range"][0]) + alpha_info["range"][0]
        if "data" in beta_info:
            beta = torch.tensor(beta_info["data"], dtype=DTYPE_MAP[beta_info["dtype"]]).reshape(beta_info["shape"])
        else:
            beta = torch.rand(beta_info["shape"], dtype=DTYPE_MAP[beta_info["dtype"]]) * (beta_info["range"][1] - beta_info["range"][0]) + beta_info["range"][0]

        input_groups.append([a, x, y, alpha, beta])
    return input_groups


def get_init_inputs():
    return []
