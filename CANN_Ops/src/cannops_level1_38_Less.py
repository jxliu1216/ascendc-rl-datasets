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

class Model(nn.Module):
    """
    实现Less算子功能的模型。
    """

    def __init__(self):
        """
        初始化模型。
        """
        super(Model, self).__init__()

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        实现Less算子功能。

        Args:
            x1: 第一个输入张量
            x2: 第二个输入张量

        Returns:
            两个输入张量逐元素比较的结果张量
        """
        return torch.less(x1, x2)

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level1_38_Less.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        x1_info = inputs[0]
        x2_info = inputs[1]

        if "data" in x1_info:
            x1 = torch.tensor(x1_info["data"], dtype=DTYPE_MAP[x1_info["dtype"]]).reshape(x1_info["shape"])
        else:
            _dt = DTYPE_MAP[x1_info["dtype"]]
            if _dt in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
                x1 = torch.randint(x1_info["range"][0], x1_info["range"][1] + 1, tuple(x1_info["shape"]), dtype=_dt)
            elif _dt == torch.bool:
                x1 = torch.rand(x1_info["shape"]) > 0.5
            else:
                x1 = torch.randn(x1_info["shape"], dtype=_dt)
        if "data" in x2_info:
            x2 = torch.tensor(x2_info["data"], dtype=DTYPE_MAP[x2_info["dtype"]]).reshape(x2_info["shape"])
        else:
            _dt = DTYPE_MAP[x2_info["dtype"]]
            if _dt in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
                x2 = torch.randint(x2_info["range"][0], x2_info["range"][1] + 1, tuple(x2_info["shape"]), dtype=_dt)
            elif _dt == torch.bool:
                x2 = torch.rand(x2_info["shape"]) > 0.5
            else:
                x2 = torch.randn(x2_info["shape"], dtype=_dt)

        input_groups.append([x1, x2])
    return input_groups


def get_init_inputs():
    return []
