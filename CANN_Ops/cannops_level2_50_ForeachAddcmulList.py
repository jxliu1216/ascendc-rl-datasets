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
    实现ForeachAddcmulList算子功能的模型。
    """

    def __init__(self):
        """
        初始化模型。
        """
        super(Model, self).__init__()

    def forward(self, x1: List[torch.Tensor], x2: List[torch.Tensor], x3: List[torch.Tensor], scalars: torch.Tensor) -> List[torch.Tensor]:
        """
        实现 ForeachAddcmulList 算子功能。

        步骤：
        1. 先把 x1、x2、x3 中的每个张量提升到 float32；
        2. scalars 保持原 dtype 不变；
        3. 执行计算：x + scalar * y * z；
        4. 返回 float32 结果列表。
        """
        ret = []
        for scalar, x, y, z in zip(scalars, x1, x2, x3):
            if x.dtype == torch.bfloat16:
                ret.append(x.to(torch.float32) + scalar.to(torch.float32) * y.to(torch.float32) * z.to(torch.float32))
            else:
                ret.append(x + scalar * y * z)
        return ret

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level2_50_ForeachAddcmulList.json')
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
            if "data" in {"dtype": x1_info["dtype"], "shape": _shape, "range": x1_info.get("range", [0, 1]), "mean": x1_info.get("mean", 0.0), "std": x1_info.get("std", 1.0), "value": x1_info.get("value")}:
                _t = torch.tensor({"dtype": x1_info["dtype"], "shape": _shape, "range": x1_info.get("range", [0, 1]), "mean": x1_info.get("mean", 0.0), "std": x1_info.get("std", 1.0), "value": x1_info.get("value")}["data"], dtype=DTYPE_MAP[{"dtype": x1_info["dtype"], "shape": _shape, "range": x1_info.get("range", [0, 1]), "mean": x1_info.get("mean", 0.0), "std": x1_info.get("std", 1.0), "value": x1_info.get("value")}["dtype"]]).reshape({"dtype": x1_info["dtype"], "shape": _shape, "range": x1_info.get("range", [0, 1]), "mean": x1_info.get("mean", 0.0), "std": x1_info.get("std", 1.0), "value": x1_info.get("value")}["shape"])
            else:
                _dt = DTYPE_MAP[{"dtype": x1_info["dtype"], "shape": _shape, "range": x1_info.get("range", [0, 1]), "mean": x1_info.get("mean", 0.0), "std": x1_info.get("std", 1.0), "value": x1_info.get("value")}["dtype"]]
                if _dt in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
                    _t = torch.randint({"dtype": x1_info["dtype"], "shape": _shape, "range": x1_info.get("range", [0, 1]), "mean": x1_info.get("mean", 0.0), "std": x1_info.get("std", 1.0), "value": x1_info.get("value")}["range"][0], {"dtype": x1_info["dtype"], "shape": _shape, "range": x1_info.get("range", [0, 1]), "mean": x1_info.get("mean", 0.0), "std": x1_info.get("std", 1.0), "value": x1_info.get("value")}["range"][1] + 1, tuple({"dtype": x1_info["dtype"], "shape": _shape, "range": x1_info.get("range", [0, 1]), "mean": x1_info.get("mean", 0.0), "std": x1_info.get("std", 1.0), "value": x1_info.get("value")}["shape"]), dtype=_dt)
                elif _dt == torch.bool:
                    _t = torch.rand({"dtype": x1_info["dtype"], "shape": _shape, "range": x1_info.get("range", [0, 1]), "mean": x1_info.get("mean", 0.0), "std": x1_info.get("std", 1.0), "value": x1_info.get("value")}["shape"]) > 0.5
                else:
                    _t = torch.rand({"dtype": x1_info["dtype"], "shape": _shape, "range": x1_info.get("range", [0, 1]), "mean": x1_info.get("mean", 0.0), "std": x1_info.get("std", 1.0), "value": x1_info.get("value")}["shape"], dtype=_dt)
            x1.append(_t)
        x2 = []
        for _shape in x2_info["shapes"]:
            if "data" in {"dtype": x2_info["dtype"], "shape": _shape, "range": x2_info.get("range", [0, 1]), "mean": x2_info.get("mean", 0.0), "std": x2_info.get("std", 1.0), "value": x2_info.get("value")}:
                _t = torch.tensor({"dtype": x2_info["dtype"], "shape": _shape, "range": x2_info.get("range", [0, 1]), "mean": x2_info.get("mean", 0.0), "std": x2_info.get("std", 1.0), "value": x2_info.get("value")}["data"], dtype=DTYPE_MAP[{"dtype": x2_info["dtype"], "shape": _shape, "range": x2_info.get("range", [0, 1]), "mean": x2_info.get("mean", 0.0), "std": x2_info.get("std", 1.0), "value": x2_info.get("value")}["dtype"]]).reshape({"dtype": x2_info["dtype"], "shape": _shape, "range": x2_info.get("range", [0, 1]), "mean": x2_info.get("mean", 0.0), "std": x2_info.get("std", 1.0), "value": x2_info.get("value")}["shape"])
            else:
                _dt = DTYPE_MAP[{"dtype": x2_info["dtype"], "shape": _shape, "range": x2_info.get("range", [0, 1]), "mean": x2_info.get("mean", 0.0), "std": x2_info.get("std", 1.0), "value": x2_info.get("value")}["dtype"]]
                if _dt in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
                    _t = torch.randint({"dtype": x2_info["dtype"], "shape": _shape, "range": x2_info.get("range", [0, 1]), "mean": x2_info.get("mean", 0.0), "std": x2_info.get("std", 1.0), "value": x2_info.get("value")}["range"][0], {"dtype": x2_info["dtype"], "shape": _shape, "range": x2_info.get("range", [0, 1]), "mean": x2_info.get("mean", 0.0), "std": x2_info.get("std", 1.0), "value": x2_info.get("value")}["range"][1] + 1, tuple({"dtype": x2_info["dtype"], "shape": _shape, "range": x2_info.get("range", [0, 1]), "mean": x2_info.get("mean", 0.0), "std": x2_info.get("std", 1.0), "value": x2_info.get("value")}["shape"]), dtype=_dt)
                elif _dt == torch.bool:
                    _t = torch.rand({"dtype": x2_info["dtype"], "shape": _shape, "range": x2_info.get("range", [0, 1]), "mean": x2_info.get("mean", 0.0), "std": x2_info.get("std", 1.0), "value": x2_info.get("value")}["shape"]) > 0.5
                else:
                    _t = torch.rand({"dtype": x2_info["dtype"], "shape": _shape, "range": x2_info.get("range", [0, 1]), "mean": x2_info.get("mean", 0.0), "std": x2_info.get("std", 1.0), "value": x2_info.get("value")}["shape"], dtype=_dt)
            x2.append(_t)
        x3 = []
        for _shape in x3_info["shapes"]:
            if "data" in {"dtype": x3_info["dtype"], "shape": _shape, "range": x3_info.get("range", [0, 1]), "mean": x3_info.get("mean", 0.0), "std": x3_info.get("std", 1.0), "value": x3_info.get("value")}:
                _t = torch.tensor({"dtype": x3_info["dtype"], "shape": _shape, "range": x3_info.get("range", [0, 1]), "mean": x3_info.get("mean", 0.0), "std": x3_info.get("std", 1.0), "value": x3_info.get("value")}["data"], dtype=DTYPE_MAP[{"dtype": x3_info["dtype"], "shape": _shape, "range": x3_info.get("range", [0, 1]), "mean": x3_info.get("mean", 0.0), "std": x3_info.get("std", 1.0), "value": x3_info.get("value")}["dtype"]]).reshape({"dtype": x3_info["dtype"], "shape": _shape, "range": x3_info.get("range", [0, 1]), "mean": x3_info.get("mean", 0.0), "std": x3_info.get("std", 1.0), "value": x3_info.get("value")}["shape"])
            else:
                _dt = DTYPE_MAP[{"dtype": x3_info["dtype"], "shape": _shape, "range": x3_info.get("range", [0, 1]), "mean": x3_info.get("mean", 0.0), "std": x3_info.get("std", 1.0), "value": x3_info.get("value")}["dtype"]]
                if _dt in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
                    _t = torch.randint({"dtype": x3_info["dtype"], "shape": _shape, "range": x3_info.get("range", [0, 1]), "mean": x3_info.get("mean", 0.0), "std": x3_info.get("std", 1.0), "value": x3_info.get("value")}["range"][0], {"dtype": x3_info["dtype"], "shape": _shape, "range": x3_info.get("range", [0, 1]), "mean": x3_info.get("mean", 0.0), "std": x3_info.get("std", 1.0), "value": x3_info.get("value")}["range"][1] + 1, tuple({"dtype": x3_info["dtype"], "shape": _shape, "range": x3_info.get("range", [0, 1]), "mean": x3_info.get("mean", 0.0), "std": x3_info.get("std", 1.0), "value": x3_info.get("value")}["shape"]), dtype=_dt)
                elif _dt == torch.bool:
                    _t = torch.rand({"dtype": x3_info["dtype"], "shape": _shape, "range": x3_info.get("range", [0, 1]), "mean": x3_info.get("mean", 0.0), "std": x3_info.get("std", 1.0), "value": x3_info.get("value")}["shape"]) > 0.5
                else:
                    _t = torch.rand({"dtype": x3_info["dtype"], "shape": _shape, "range": x3_info.get("range", [0, 1]), "mean": x3_info.get("mean", 0.0), "std": x3_info.get("std", 1.0), "value": x3_info.get("value")}["shape"], dtype=_dt)
            x3.append(_t)
        if "data" in scalars_info:
            scalars = torch.tensor(scalars_info["data"], dtype=DTYPE_MAP[scalars_info["dtype"]]).reshape(scalars_info["shape"])
        else:
            _dt = DTYPE_MAP[scalars_info["dtype"]]
            if _dt in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
                scalars = torch.randint(scalars_info["range"][0], scalars_info["range"][1] + 1, tuple(scalars_info["shape"]), dtype=_dt)
            elif _dt == torch.bool:
                scalars = torch.rand(scalars_info["shape"]) > 0.5
            else:
                scalars = torch.rand(scalars_info["shape"], dtype=_dt) * (scalars_info["range"][1] - scalars_info["range"][0]) + scalars_info["range"][0]

        input_groups.append([x1, x2, x3, scalars])
    return input_groups


def get_init_inputs():
    return []
