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
    实现ForeachMinimumScalar算子功能的模型。
    """

    def __init__(self):
        """
        初始化模型。
        """
        super(Model, self).__init__()

    def forward(self, inputs: List[torch.Tensor], scalar: torch.Tensor) -> List[torch.Tensor]:
        """
        实现ForeachMinimumScalar算子功能。

        Args:
            inputs: 输入张量列表
            scalar: 标量张量

        Returns:
            输入张量列表和标量逐元素比较后的最大值张量列表
        """
        return [torch.minimum(x, scalar) for x in inputs]

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level2_75_ForeachMinimumScalar.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        inputs_info = inputs[0]
        scalar_info = inputs[1]

        inputs = []
        for _shape in inputs_info["shapes"]:
            if "data" in {"dtype": inputs_info["dtype"], "shape": _shape, "range": inputs_info.get("range", [0, 1]), "mean": inputs_info.get("mean", 0.0), "std": inputs_info.get("std", 1.0), "value": inputs_info.get("value")}:
                _t = torch.tensor({"dtype": inputs_info["dtype"], "shape": _shape, "range": inputs_info.get("range", [0, 1]), "mean": inputs_info.get("mean", 0.0), "std": inputs_info.get("std", 1.0), "value": inputs_info.get("value")}["data"], dtype=DTYPE_MAP[{"dtype": inputs_info["dtype"], "shape": _shape, "range": inputs_info.get("range", [0, 1]), "mean": inputs_info.get("mean", 0.0), "std": inputs_info.get("std", 1.0), "value": inputs_info.get("value")}["dtype"]]).reshape({"dtype": inputs_info["dtype"], "shape": _shape, "range": inputs_info.get("range", [0, 1]), "mean": inputs_info.get("mean", 0.0), "std": inputs_info.get("std", 1.0), "value": inputs_info.get("value")}["shape"])
            else:
                _dt = DTYPE_MAP[{"dtype": inputs_info["dtype"], "shape": _shape, "range": inputs_info.get("range", [0, 1]), "mean": inputs_info.get("mean", 0.0), "std": inputs_info.get("std", 1.0), "value": inputs_info.get("value")}["dtype"]]
                if _dt in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
                    _t = torch.randint({"dtype": inputs_info["dtype"], "shape": _shape, "range": inputs_info.get("range", [0, 1]), "mean": inputs_info.get("mean", 0.0), "std": inputs_info.get("std", 1.0), "value": inputs_info.get("value")}["range"][0], {"dtype": inputs_info["dtype"], "shape": _shape, "range": inputs_info.get("range", [0, 1]), "mean": inputs_info.get("mean", 0.0), "std": inputs_info.get("std", 1.0), "value": inputs_info.get("value")}["range"][1] + 1, tuple({"dtype": inputs_info["dtype"], "shape": _shape, "range": inputs_info.get("range", [0, 1]), "mean": inputs_info.get("mean", 0.0), "std": inputs_info.get("std", 1.0), "value": inputs_info.get("value")}["shape"]), dtype=_dt)
                elif _dt == torch.bool:
                    _t = torch.rand({"dtype": inputs_info["dtype"], "shape": _shape, "range": inputs_info.get("range", [0, 1]), "mean": inputs_info.get("mean", 0.0), "std": inputs_info.get("std", 1.0), "value": inputs_info.get("value")}["shape"]) > 0.5
                else:
                    _t = torch.rand({"dtype": inputs_info["dtype"], "shape": _shape, "range": inputs_info.get("range", [0, 1]), "mean": inputs_info.get("mean", 0.0), "std": inputs_info.get("std", 1.0), "value": inputs_info.get("value")}["shape"], dtype=_dt) * ({"dtype": inputs_info["dtype"], "shape": _shape, "range": inputs_info.get("range", [0, 1]), "mean": inputs_info.get("mean", 0.0), "std": inputs_info.get("std", 1.0), "value": inputs_info.get("value")}["range"][1] - {"dtype": inputs_info["dtype"], "shape": _shape, "range": inputs_info.get("range", [0, 1]), "mean": inputs_info.get("mean", 0.0), "std": inputs_info.get("std", 1.0), "value": inputs_info.get("value")}["range"][0]) + {"dtype": inputs_info["dtype"], "shape": _shape, "range": inputs_info.get("range", [0, 1]), "mean": inputs_info.get("mean", 0.0), "std": inputs_info.get("std", 1.0), "value": inputs_info.get("value")}["range"][0]
            inputs.append(_t)
        if "data" in scalar_info:
            scalar = torch.tensor(scalar_info["data"], dtype=DTYPE_MAP[scalar_info["dtype"]]).reshape(scalar_info["shape"])
        else:
            _dt = DTYPE_MAP[scalar_info["dtype"]]
            if _dt in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
                scalar = torch.full(scalar_info["shape"], scalar_info["fill"], dtype=_dt)
            elif _dt == torch.bool:
                scalar = torch.rand(scalar_info["shape"]) > 0.5
            else:
                scalar = torch.rand(scalar_info["shape"], dtype=_dt) * (scalar_info["range"][1] - scalar_info["range"][0]) + scalar_info["range"][0]

        input_groups.append([inputs, scalar])
    return input_groups


def get_init_inputs():
    return []
