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
    实现ForeachDivScalarList算子功能的模型。
    """

    def __init__(self):
        """
        初始化模型。
        """
        super(Model, self).__init__()

    def forward(self, x: List[torch.Tensor], scalars: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        实现ForeachDivScalarList算子功能。

        Args:
            x: 输入张量列表
            scalars: 标量张量列表

        Returns:
            输入张量列表与标量张量列表逐元素相除后的结果张量列表
        """
        return [tensor / scalar for tensor, scalar in zip(x, scalars)]

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level2_60_ForeachDivScalarList.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        x_info = inputs[0]
        scalars_info = inputs[1]

        x = []
        for _shape in x_info["shapes"]:
            _t = torch.rand({"dtype": x_info["dtype"], "shape": _shape, "range": x_info.get("range", [0, 1]), "mean": x_info.get("mean", 0.0), "std": x_info.get("std", 1.0), "value": x_info.get("value")}["shape"], dtype=DTYPE_MAP[{"dtype": x_info["dtype"], "shape": _shape, "range": x_info.get("range", [0, 1]), "mean": x_info.get("mean", 0.0), "std": x_info.get("std", 1.0), "value": x_info.get("value")}["dtype"]]) * ({"dtype": x_info["dtype"], "shape": _shape, "range": x_info.get("range", [0, 1]), "mean": x_info.get("mean", 0.0), "std": x_info.get("std", 1.0), "value": x_info.get("value")}["range"][1] - {"dtype": x_info["dtype"], "shape": _shape, "range": x_info.get("range", [0, 1]), "mean": x_info.get("mean", 0.0), "std": x_info.get("std", 1.0), "value": x_info.get("value")}["range"][0]) + {"dtype": x_info["dtype"], "shape": _shape, "range": x_info.get("range", [0, 1]), "mean": x_info.get("mean", 0.0), "std": x_info.get("std", 1.0), "value": x_info.get("value")}["range"][0]
            x.append(_t)
        if "data" in scalars_info:
            scalars = torch.tensor(scalars_info["data"], dtype=DTYPE_MAP[scalars_info["dtype"]]).reshape(scalars_info["shape"])
        else:
            scalars = torch.rand(scalars_info["shape"], dtype=DTYPE_MAP[scalars_info["dtype"]]) * (scalars_info["range"][1] - scalars_info["range"][0]) + scalars_info["range"][0]

        input_groups.append([x, scalars])
    return input_groups


def get_init_inputs():
    return []
