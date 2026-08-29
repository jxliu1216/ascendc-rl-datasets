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
    实现概率到对数几率转换（log-odds)。
    """

    def __init__(self):
        """
        初始化模型。

        Args:
        """
        super(Model, self).__init__()

    def forward(self, x_tensor, dy_tensor, eps=None):
        """
        使用torch.special.logit + backward功能实现，获取标杆结果。

        Args:
            x_tensor: 输入张量，数值在(0-1)之间
            dy_tensor: 正向输出结果的梯度
            eps: double 输入x_tensor的epsilon限制边界，建议值-1

        Returns:
            out: dx, 输出张量，与输入数据类型、shape与输入x_tensor一致
        """
        x_tensor = x_tensor.requires_grad_(True)
        y = torch.special.logit(x_tensor, eps)
        y.backward(dy_tensor)
        return x_tensor.grad

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level2_122_LogitGrad.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        x_tensor_info = inputs[0]
        dy_tensor_info = inputs[1]
        eps_info = inputs[2]

        if "data" in x_tensor_info:
            x_tensor = torch.tensor(x_tensor_info["data"], dtype=DTYPE_MAP[x_tensor_info["dtype"]]).reshape(x_tensor_info["shape"])
        else:
            x_tensor = torch.rand(x_tensor_info["shape"], dtype=DTYPE_MAP[x_tensor_info["dtype"]])
        if "data" in dy_tensor_info:
            dy_tensor = torch.tensor(dy_tensor_info["data"], dtype=DTYPE_MAP[dy_tensor_info["dtype"]]).reshape(dy_tensor_info["shape"])
        else:
            dy_tensor = torch.rand(dy_tensor_info["shape"], dtype=DTYPE_MAP[dy_tensor_info["dtype"]])
        if eps_info["type"] == "attr":
            if eps_info.get("dtype") == "none":
                eps = None
            else:
                eps = eps_info["value"]

        input_groups.append([x_tensor, dy_tensor, eps])
    return input_groups


def get_init_inputs():
    return []
