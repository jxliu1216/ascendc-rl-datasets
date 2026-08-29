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
    Simple model that performs log operation.
    """

    def __init__(self):
        """
        Initialize the model for log operation.
        No parameters needed for basic log operation.
        """
        super(Model, self).__init__()

    def forward(self, x: List[torch.Tensor], alpha: torch.Tensor) -> List[torch.Tensor]:
        """
        逐张量计算：out_i = x_i + α_i
        α 是一维 Tensor，长度等于 len(x)。
        中间值用 float32，结果回到原 dtype。
        """
        return [(t.float() + alpha[i]).to(t.dtype) for i, t in enumerate(x)]

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level2_46_ForeachAddScalarList.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        x_info = inputs[0]
        alpha_info = inputs[1]

        x = []
        for _shape in x_info["shapes"]:
            if "data" in {"dtype": x_info["dtype"], "shape": _shape, "range": x_info.get("range", [0, 1]), "mean": x_info.get("mean", 0.0), "std": x_info.get("std", 1.0), "value": x_info.get("value")}:
                _t = torch.tensor({"dtype": x_info["dtype"], "shape": _shape, "range": x_info.get("range", [0, 1]), "mean": x_info.get("mean", 0.0), "std": x_info.get("std", 1.0), "value": x_info.get("value")}["data"], dtype=DTYPE_MAP[{"dtype": x_info["dtype"], "shape": _shape, "range": x_info.get("range", [0, 1]), "mean": x_info.get("mean", 0.0), "std": x_info.get("std", 1.0), "value": x_info.get("value")}["dtype"]]).reshape({"dtype": x_info["dtype"], "shape": _shape, "range": x_info.get("range", [0, 1]), "mean": x_info.get("mean", 0.0), "std": x_info.get("std", 1.0), "value": x_info.get("value")}["shape"])
            else:
                _dt = DTYPE_MAP[{"dtype": x_info["dtype"], "shape": _shape, "range": x_info.get("range", [0, 1]), "mean": x_info.get("mean", 0.0), "std": x_info.get("std", 1.0), "value": x_info.get("value")}["dtype"]]
                if _dt in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
                    _t = torch.randint({"dtype": x_info["dtype"], "shape": _shape, "range": x_info.get("range", [0, 1]), "mean": x_info.get("mean", 0.0), "std": x_info.get("std", 1.0), "value": x_info.get("value")}["range"][0], {"dtype": x_info["dtype"], "shape": _shape, "range": x_info.get("range", [0, 1]), "mean": x_info.get("mean", 0.0), "std": x_info.get("std", 1.0), "value": x_info.get("value")}["range"][1] + 1, tuple({"dtype": x_info["dtype"], "shape": _shape, "range": x_info.get("range", [0, 1]), "mean": x_info.get("mean", 0.0), "std": x_info.get("std", 1.0), "value": x_info.get("value")}["shape"]), dtype=_dt)
                elif _dt == torch.bool:
                    _t = torch.rand({"dtype": x_info["dtype"], "shape": _shape, "range": x_info.get("range", [0, 1]), "mean": x_info.get("mean", 0.0), "std": x_info.get("std", 1.0), "value": x_info.get("value")}["shape"]) > 0.5
                else:
                    _t = torch.rand({"dtype": x_info["dtype"], "shape": _shape, "range": x_info.get("range", [0, 1]), "mean": x_info.get("mean", 0.0), "std": x_info.get("std", 1.0), "value": x_info.get("value")}["shape"], dtype=_dt)
            x.append(_t)
        if "data" in alpha_info:
            alpha = torch.tensor(alpha_info["data"], dtype=DTYPE_MAP[alpha_info["dtype"]]).reshape(alpha_info["shape"])
        else:
            _dt = DTYPE_MAP[alpha_info["dtype"]]
            if _dt in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
                alpha = torch.randint(alpha_info["range"][0], alpha_info["range"][1] + 1, tuple(alpha_info["shape"]), dtype=_dt)
            elif _dt == torch.bool:
                alpha = torch.rand(alpha_info["shape"]) > 0.5
            else:
                alpha = torch.rand(alpha_info["shape"], dtype=_dt) * (alpha_info["range"][1] - alpha_info["range"][0]) + alpha_info["range"][0]

        input_groups.append([x, alpha])
    return input_groups


def get_init_inputs():
    return []
