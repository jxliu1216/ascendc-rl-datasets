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

import torch
import torch.nn as nn
import torch.nn.functional as F
import ast

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, tensor_dy: torch.Tensor, tensor_x: torch.Tensor, dim: int, approximate_int: int, activateLeft: bool) -> torch.Tensor:
        """
        实现 GeGLUGradV2 的前向和梯度计算。
        此方法旨在计算 y 对 tensor_x 的梯度，给定上游梯度 tensor_dy。

        Args:
            tensor_dy (torch.Tensor): 上游传来的梯度，形状应与 y 相同。
            tensor_x (torch.Tensor): 输入张量，将被分割并用于计算 GeLU。
            gelu_output (torch.Tensor): 原始代码中的占位符，修改后不再用于输出。
            dim (int): 用于 chunk 操作的维度。
            approximate_int (int): GeLU 函数的近似模式：0 表示 'none'/'erf'，1 表示 'tanh'。
            activateLeft (bool): 未在当前实现中使用。

        Returns:
            torch.Tensor: tensor_x 的梯度。
        """
        approximate_map = {0: 'none', 1: 'tanh'}
        approximate_str = approximate_map.get(approximate_int, 'none')
        with torch.enable_grad():
            x_chunk, gate_chunk = torch.chunk(tensor_x, 2, dim=dim)
            x_for_mul, gate_for_gelu = (gate_chunk, x_chunk)
            y_gelu = F.gelu(gate_for_gelu, approximate=approximate_str)
            y = x_for_mul * y_gelu
            grad_tensor_x = torch.autograd.grad(outputs=y, inputs=tensor_x, grad_outputs=tensor_dy)[0]
        return grad_tensor_x

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level2_100_GeGluGradV2.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        tensor_dy_info = inputs[0]
        tensor_x_info = inputs[1]
        dim_info = inputs[2]
        approximate_int_info = inputs[3]
        activateLeft_info = inputs[4]

        if "data" in tensor_dy_info:
            tensor_dy = torch.tensor(tensor_dy_info["data"], dtype=DTYPE_MAP[tensor_dy_info["dtype"]]).reshape(tensor_dy_info["shape"])
        else:
            tensor_dy = torch.rand(tensor_dy_info["shape"], dtype=DTYPE_MAP[tensor_dy_info["dtype"]])
        if "data" in tensor_x_info:
            tensor_x = torch.tensor(tensor_x_info["data"], dtype=DTYPE_MAP[tensor_x_info["dtype"]]).reshape(tensor_x_info["shape"])
        else:
            tensor_x = torch.rand(tensor_x_info["shape"], dtype=DTYPE_MAP[tensor_x_info["dtype"]])
        dim = dim_info["value"]
        approximate_int = approximate_int_info["value"]
        activateLeft = activateLeft_info["value"]

        input_groups.append([tensor_dy, tensor_x, dim, approximate_int, activateLeft])
    return input_groups


def get_init_inputs():
    return []
