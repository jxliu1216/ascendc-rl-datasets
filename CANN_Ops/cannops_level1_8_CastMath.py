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
    实现Cast算子功能的模型。
    """

    def __init__(self):
        """
        初始化模型。
        """
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, dst_type: int) -> torch.Tensor:
        """
        实现Cast算子功能。

        Args:
            x: 输入张量
            dst_type: 目标数据类型的标识

        Returns:
            转换为目标数据类型后的张量
        """
        return x.to(self._get_dtype(dst_type))

    def _get_dtype(self, dst_type):
        type_map = {0: torch.float32, 1: torch.float16, 2: torch.int8, 3: torch.int32, 4: torch.uint8, 6: torch.int16, 9: torch.int64, 12: torch.bool, 27: torch.bfloat16}
        return type_map.get(dst_type, torch.float32)

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level1_8_CastMath.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        x_info = inputs[0]
        dst_type_info = inputs[1]

        if "data" in x_info:
            x = torch.tensor(x_info["data"], dtype=DTYPE_MAP[x_info["dtype"]]).reshape(x_info["shape"])
        else:
            _dt = DTYPE_MAP[x_info["dtype"]]
            if _dt in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
                x = torch.randint(x_info["range"][0], x_info["range"][1] + 1, tuple(x_info["shape"]), dtype=_dt)
            elif _dt == torch.bool:
                x = torch.rand(x_info["shape"]) > 0.5
            else:
                x = torch.randn(x_info["shape"], dtype=_dt)
        dst_type = dst_type_info["value"]

        input_groups.append([x, dst_type])
    return input_groups


def get_init_inputs():
    return []
