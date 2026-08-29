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
    简单模型，对输入张量列表中的每个张量与对应标量相乘。
    """

    def __init__(self):
        """
        初始化模型。
        标量乘法操作不需要额外参数。
        """
        super(Model, self).__init__()

    def forward(self, inputs: List[torch.Tensor], scalar_list) -> List[torch.Tensor]:
        """
        计算每个输入张量与对应标量的乘积。

        Args:
            inputs: 输入张量列表，可以是任意形状。
            scalar_list: 标量列表，用于与输入张量相乘。

        Returns:
            与输入张量具有相同形状的输出张量列表，每个输出是输入张量与对应标量的乘积。
        """
        return [a * x for a, x in zip(scalar_list, inputs)]

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level2_79_ForeachMulScalarList.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        inputs_info = inputs[0]
        scalar_list_info = inputs[1]

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
        if "data" in scalar_list_info:
            scalar_list = torch.tensor(scalar_list_info["data"], dtype=DTYPE_MAP[scalar_list_info["dtype"]]).reshape(scalar_list_info["shape"])
        else:
            _dt = DTYPE_MAP[scalar_list_info["dtype"]]
            if _dt in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
                scalar_list = torch.randint(scalar_list_info["range"][0], scalar_list_info["range"][1] + 1, tuple(scalar_list_info["shape"]), dtype=_dt)
            elif _dt == torch.bool:
                scalar_list = torch.rand(scalar_list_info["shape"]) > 0.5
            else:
                scalar_list = torch.rand(scalar_list_info["shape"], dtype=_dt) * (scalar_list_info["range"][1] - scalar_list_info["range"][0]) + scalar_list_info["range"][0]

        input_groups.append([inputs, scalar_list])
    return input_groups


def get_init_inputs():
    return []
