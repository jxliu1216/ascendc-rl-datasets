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
    实现ForeachAddcdivList算子功能的模型。
    """

    def __init__(self):
        """
        初始化模型。
        """
        super(Model, self).__init__()

    def forward(self, x1: List[torch.Tensor], x2: List[torch.Tensor], x3: List[torch.Tensor], scalars: torch.Tensor) -> List[torch.Tensor]:
        """
        实现ForeachAddcdivList算子功能。

        Args:
            x1: 第一个输入张量列表
            x2: 第二个输入张量列表
            x3: 第三个输入张量列表
            scalars: 标量张量

        Returns:
            经过逐元素加、乘、除操作后的结果张量列表
        """
        return [x + y / z * s for x, y, z, s in zip(x1, x2, x3, scalars)]

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level2_47_ForeachAddcdivList.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        x1_info = inputs[0]
        x2_info = inputs[1]
        x3_info = inputs[2]
        scalars_info = inputs[3]

        x1 = []
        for _shape in x1_info["shapes"]:
            _t = torch.rand({"dtype": x1_info["dtype"], "shape": _shape, "range": x1_info.get("range", [0, 1]), "mean": x1_info.get("mean", 0.0), "std": x1_info.get("std", 1.0), "value": x1_info.get("value")}["shape"], dtype=DTYPE_MAP[{"dtype": x1_info["dtype"], "shape": _shape, "range": x1_info.get("range", [0, 1]), "mean": x1_info.get("mean", 0.0), "std": x1_info.get("std", 1.0), "value": x1_info.get("value")}["dtype"]]) * ({"dtype": x1_info["dtype"], "shape": _shape, "range": x1_info.get("range", [0, 1]), "mean": x1_info.get("mean", 0.0), "std": x1_info.get("std", 1.0), "value": x1_info.get("value")}["range"][1] - {"dtype": x1_info["dtype"], "shape": _shape, "range": x1_info.get("range", [0, 1]), "mean": x1_info.get("mean", 0.0), "std": x1_info.get("std", 1.0), "value": x1_info.get("value")}["range"][0]) + {"dtype": x1_info["dtype"], "shape": _shape, "range": x1_info.get("range", [0, 1]), "mean": x1_info.get("mean", 0.0), "std": x1_info.get("std", 1.0), "value": x1_info.get("value")}["range"][0]
            x1.append(_t)
        x2 = []
        for _shape in x2_info["shapes"]:
            _t = torch.rand({"dtype": x2_info["dtype"], "shape": _shape, "range": x2_info.get("range", [0, 1]), "mean": x2_info.get("mean", 0.0), "std": x2_info.get("std", 1.0), "value": x2_info.get("value")}["shape"], dtype=DTYPE_MAP[{"dtype": x2_info["dtype"], "shape": _shape, "range": x2_info.get("range", [0, 1]), "mean": x2_info.get("mean", 0.0), "std": x2_info.get("std", 1.0), "value": x2_info.get("value")}["dtype"]]) * ({"dtype": x2_info["dtype"], "shape": _shape, "range": x2_info.get("range", [0, 1]), "mean": x2_info.get("mean", 0.0), "std": x2_info.get("std", 1.0), "value": x2_info.get("value")}["range"][1] - {"dtype": x2_info["dtype"], "shape": _shape, "range": x2_info.get("range", [0, 1]), "mean": x2_info.get("mean", 0.0), "std": x2_info.get("std", 1.0), "value": x2_info.get("value")}["range"][0]) + {"dtype": x2_info["dtype"], "shape": _shape, "range": x2_info.get("range", [0, 1]), "mean": x2_info.get("mean", 0.0), "std": x2_info.get("std", 1.0), "value": x2_info.get("value")}["range"][0]
            x2.append(_t)
        x3 = []
        for _shape in x3_info["shapes"]:
            _t = torch.rand({"dtype": x3_info["dtype"], "shape": _shape, "range": x3_info.get("range", [0, 1]), "mean": x3_info.get("mean", 0.0), "std": x3_info.get("std", 1.0), "value": x3_info.get("value")}["shape"], dtype=DTYPE_MAP[{"dtype": x3_info["dtype"], "shape": _shape, "range": x3_info.get("range", [0, 1]), "mean": x3_info.get("mean", 0.0), "std": x3_info.get("std", 1.0), "value": x3_info.get("value")}["dtype"]]) * ({"dtype": x3_info["dtype"], "shape": _shape, "range": x3_info.get("range", [0, 1]), "mean": x3_info.get("mean", 0.0), "std": x3_info.get("std", 1.0), "value": x3_info.get("value")}["range"][1] - {"dtype": x3_info["dtype"], "shape": _shape, "range": x3_info.get("range", [0, 1]), "mean": x3_info.get("mean", 0.0), "std": x3_info.get("std", 1.0), "value": x3_info.get("value")}["range"][0]) + {"dtype": x3_info["dtype"], "shape": _shape, "range": x3_info.get("range", [0, 1]), "mean": x3_info.get("mean", 0.0), "std": x3_info.get("std", 1.0), "value": x3_info.get("value")}["range"][0]
            x3.append(_t)
        if "data" in scalars_info:
            scalars = torch.tensor(scalars_info["data"], dtype=DTYPE_MAP[scalars_info["dtype"]]).reshape(scalars_info["shape"])
        else:
            scalars = torch.rand(scalars_info["shape"], dtype=DTYPE_MAP[scalars_info["dtype"]]) * (scalars_info["range"][1] - scalars_info["range"][0]) + scalars_info["range"][0]

        input_groups.append([x1, x2, x3, scalars])
    return input_groups


def get_init_inputs():
    return []
