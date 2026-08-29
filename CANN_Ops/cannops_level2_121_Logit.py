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

    def forward(self, input_tensor, eps=None):
        """
        使用torch.logit功能实现，获取标杆结果。

        Args:
            input_tensor: 输入张量
            eps: double 输入input的epsilon限制边界，建议值-1

        Returns:
            out: 转换后结果张量
        """
        return torch.logit(input_tensor, eps)

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level2_121_Logit.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        input_tensor_info = inputs[0]
        eps_info = inputs[1]

        if "data" in input_tensor_info:
            input_tensor = torch.tensor(input_tensor_info["data"], dtype=DTYPE_MAP[input_tensor_info["dtype"]]).reshape(input_tensor_info["shape"])
        else:
            input_tensor = torch.rand(input_tensor_info["shape"], dtype=DTYPE_MAP[input_tensor_info["dtype"]])
        if eps_info["type"] == "attr":
            if eps_info.get("dtype") == "none":
                eps = None
            else:
                eps = eps_info["value"]

        input_groups.append([input_tensor, eps])
    return input_groups


def get_init_inputs():
    return []
