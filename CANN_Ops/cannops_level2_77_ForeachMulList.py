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
    简单模型，对两个输入张量列表中对应位置的张量进行相乘操作。
    """

    def __init__(self):
        """
        初始化模型。
        张量乘法操作不需要额外参数。
        """
        super(Model, self).__init__()

    def forward(self, inputs1: List[torch.Tensor], inputs2: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        计算两个张量列表中对应位置张量的乘积。

        Args:
            inputs1: 第一个输入张量列表，可以是任意形状。
            inputs2: 第二个输入张量列表，可以是任意形状。

        Returns:
            与输入张量具有相同形状的输出张量列表，每个输出是对应位置两个输入张量的乘积。
        """
        return [torch.mul(x, y) for x, y in zip(inputs1, inputs2)]

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level2_77_ForeachMulList.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        inputs1_info = inputs[0]
        inputs2_info = inputs[1]

        inputs1 = []
        for _shape in inputs1_info["shapes"]:
            if "data" in {"dtype": inputs1_info["dtype"], "shape": _shape, "range": inputs1_info.get("range", [0, 1]), "mean": inputs1_info.get("mean", 0.0), "std": inputs1_info.get("std", 1.0), "value": inputs1_info.get("value")}:
                _t = torch.tensor({"dtype": inputs1_info["dtype"], "shape": _shape, "range": inputs1_info.get("range", [0, 1]), "mean": inputs1_info.get("mean", 0.0), "std": inputs1_info.get("std", 1.0), "value": inputs1_info.get("value")}["data"], dtype=DTYPE_MAP[{"dtype": inputs1_info["dtype"], "shape": _shape, "range": inputs1_info.get("range", [0, 1]), "mean": inputs1_info.get("mean", 0.0), "std": inputs1_info.get("std", 1.0), "value": inputs1_info.get("value")}["dtype"]]).reshape({"dtype": inputs1_info["dtype"], "shape": _shape, "range": inputs1_info.get("range", [0, 1]), "mean": inputs1_info.get("mean", 0.0), "std": inputs1_info.get("std", 1.0), "value": inputs1_info.get("value")}["shape"])
            else:
                _dt = DTYPE_MAP[{"dtype": inputs1_info["dtype"], "shape": _shape, "range": inputs1_info.get("range", [0, 1]), "mean": inputs1_info.get("mean", 0.0), "std": inputs1_info.get("std", 1.0), "value": inputs1_info.get("value")}["dtype"]]
                if _dt in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
                    _t = torch.randint({"dtype": inputs1_info["dtype"], "shape": _shape, "range": inputs1_info.get("range", [0, 1]), "mean": inputs1_info.get("mean", 0.0), "std": inputs1_info.get("std", 1.0), "value": inputs1_info.get("value")}["range"][0], {"dtype": inputs1_info["dtype"], "shape": _shape, "range": inputs1_info.get("range", [0, 1]), "mean": inputs1_info.get("mean", 0.0), "std": inputs1_info.get("std", 1.0), "value": inputs1_info.get("value")}["range"][1] + 1, tuple({"dtype": inputs1_info["dtype"], "shape": _shape, "range": inputs1_info.get("range", [0, 1]), "mean": inputs1_info.get("mean", 0.0), "std": inputs1_info.get("std", 1.0), "value": inputs1_info.get("value")}["shape"]), dtype=_dt)
                elif _dt == torch.bool:
                    _t = torch.rand({"dtype": inputs1_info["dtype"], "shape": _shape, "range": inputs1_info.get("range", [0, 1]), "mean": inputs1_info.get("mean", 0.0), "std": inputs1_info.get("std", 1.0), "value": inputs1_info.get("value")}["shape"]) > 0.5
                else:
                    _t = torch.randn({"dtype": inputs1_info["dtype"], "shape": _shape, "range": inputs1_info.get("range", [0, 1]), "mean": inputs1_info.get("mean", 0.0), "std": inputs1_info.get("std", 1.0), "value": inputs1_info.get("value")}["shape"], dtype=_dt)
            inputs1.append(_t)
        inputs2 = []
        for _shape in inputs2_info["shapes"]:
            if "data" in {"dtype": inputs2_info["dtype"], "shape": _shape, "range": inputs2_info.get("range", [0, 1]), "mean": inputs2_info.get("mean", 0.0), "std": inputs2_info.get("std", 1.0), "value": inputs2_info.get("value")}:
                _t = torch.tensor({"dtype": inputs2_info["dtype"], "shape": _shape, "range": inputs2_info.get("range", [0, 1]), "mean": inputs2_info.get("mean", 0.0), "std": inputs2_info.get("std", 1.0), "value": inputs2_info.get("value")}["data"], dtype=DTYPE_MAP[{"dtype": inputs2_info["dtype"], "shape": _shape, "range": inputs2_info.get("range", [0, 1]), "mean": inputs2_info.get("mean", 0.0), "std": inputs2_info.get("std", 1.0), "value": inputs2_info.get("value")}["dtype"]]).reshape({"dtype": inputs2_info["dtype"], "shape": _shape, "range": inputs2_info.get("range", [0, 1]), "mean": inputs2_info.get("mean", 0.0), "std": inputs2_info.get("std", 1.0), "value": inputs2_info.get("value")}["shape"])
            else:
                _dt = DTYPE_MAP[{"dtype": inputs2_info["dtype"], "shape": _shape, "range": inputs2_info.get("range", [0, 1]), "mean": inputs2_info.get("mean", 0.0), "std": inputs2_info.get("std", 1.0), "value": inputs2_info.get("value")}["dtype"]]
                if _dt in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
                    _t = torch.randint({"dtype": inputs2_info["dtype"], "shape": _shape, "range": inputs2_info.get("range", [0, 1]), "mean": inputs2_info.get("mean", 0.0), "std": inputs2_info.get("std", 1.0), "value": inputs2_info.get("value")}["range"][0], {"dtype": inputs2_info["dtype"], "shape": _shape, "range": inputs2_info.get("range", [0, 1]), "mean": inputs2_info.get("mean", 0.0), "std": inputs2_info.get("std", 1.0), "value": inputs2_info.get("value")}["range"][1] + 1, tuple({"dtype": inputs2_info["dtype"], "shape": _shape, "range": inputs2_info.get("range", [0, 1]), "mean": inputs2_info.get("mean", 0.0), "std": inputs2_info.get("std", 1.0), "value": inputs2_info.get("value")}["shape"]), dtype=_dt)
                elif _dt == torch.bool:
                    _t = torch.rand({"dtype": inputs2_info["dtype"], "shape": _shape, "range": inputs2_info.get("range", [0, 1]), "mean": inputs2_info.get("mean", 0.0), "std": inputs2_info.get("std", 1.0), "value": inputs2_info.get("value")}["shape"]) > 0.5
                else:
                    _t = torch.randn({"dtype": inputs2_info["dtype"], "shape": _shape, "range": inputs2_info.get("range", [0, 1]), "mean": inputs2_info.get("mean", 0.0), "std": inputs2_info.get("std", 1.0), "value": inputs2_info.get("value")}["shape"], dtype=_dt)
            inputs2.append(_t)

        input_groups.append([inputs1, inputs2])
    return input_groups


def get_init_inputs():
    return []
