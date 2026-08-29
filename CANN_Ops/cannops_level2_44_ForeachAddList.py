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

    def __init__(self):
        super().__init__()

    def forward(self, x1: List[torch.Tensor], x2: List[torch.Tensor], alpha: torch.Tensor) -> List[torch.Tensor]:
        """
        Foreach: out[i] = x1[i] + x2[i] * alpha
        """
        out = []
        for a, b in zip(x1, x2):
            if b.dtype == torch.bfloat16:
                res = (a.to(torch.float32) + b.to(torch.float32) * alpha).to(torch.bfloat16)
            else:
                res = a + b * alpha
            out.append(res)
        return out

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level2_44_ForeachAddList.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        x1_info = inputs[0]
        x2_info = inputs[1]
        alpha_info = inputs[2]

        x1 = []
        for _shape in x1_info["shapes"]:
            _t = torch.rand({"dtype": x1_info["dtype"], "shape": _shape, "range": x1_info.get("range", [0, 1]), "mean": x1_info.get("mean", 0.0), "std": x1_info.get("std", 1.0), "value": x1_info.get("value")}["shape"], dtype=DTYPE_MAP[{"dtype": x1_info["dtype"], "shape": _shape, "range": x1_info.get("range", [0, 1]), "mean": x1_info.get("mean", 0.0), "std": x1_info.get("std", 1.0), "value": x1_info.get("value")}["dtype"]]) * ({"dtype": x1_info["dtype"], "shape": _shape, "range": x1_info.get("range", [0, 1]), "mean": x1_info.get("mean", 0.0), "std": x1_info.get("std", 1.0), "value": x1_info.get("value")}["range"][1] - {"dtype": x1_info["dtype"], "shape": _shape, "range": x1_info.get("range", [0, 1]), "mean": x1_info.get("mean", 0.0), "std": x1_info.get("std", 1.0), "value": x1_info.get("value")}["range"][0]) + {"dtype": x1_info["dtype"], "shape": _shape, "range": x1_info.get("range", [0, 1]), "mean": x1_info.get("mean", 0.0), "std": x1_info.get("std", 1.0), "value": x1_info.get("value")}["range"][0]
            x1.append(_t)
        x2 = []
        for _shape in x2_info["shapes"]:
            _t = torch.rand({"dtype": x2_info["dtype"], "shape": _shape, "range": x2_info.get("range", [0, 1]), "mean": x2_info.get("mean", 0.0), "std": x2_info.get("std", 1.0), "value": x2_info.get("value")}["shape"], dtype=DTYPE_MAP[{"dtype": x2_info["dtype"], "shape": _shape, "range": x2_info.get("range", [0, 1]), "mean": x2_info.get("mean", 0.0), "std": x2_info.get("std", 1.0), "value": x2_info.get("value")}["dtype"]]) * ({"dtype": x2_info["dtype"], "shape": _shape, "range": x2_info.get("range", [0, 1]), "mean": x2_info.get("mean", 0.0), "std": x2_info.get("std", 1.0), "value": x2_info.get("value")}["range"][1] - {"dtype": x2_info["dtype"], "shape": _shape, "range": x2_info.get("range", [0, 1]), "mean": x2_info.get("mean", 0.0), "std": x2_info.get("std", 1.0), "value": x2_info.get("value")}["range"][0]) + {"dtype": x2_info["dtype"], "shape": _shape, "range": x2_info.get("range", [0, 1]), "mean": x2_info.get("mean", 0.0), "std": x2_info.get("std", 1.0), "value": x2_info.get("value")}["range"][0]
            x2.append(_t)
        if "data" in alpha_info:
            alpha = torch.tensor(alpha_info["data"], dtype=DTYPE_MAP[alpha_info["dtype"]]).reshape(alpha_info["shape"])
        else:
            alpha = torch.rand(alpha_info["shape"], dtype=DTYPE_MAP[alpha_info["dtype"]]) * (alpha_info["range"][1] - alpha_info["range"][0]) + alpha_info["range"][0]

        input_groups.append([x1, x2, alpha])
    return input_groups


def get_init_inputs():
    return []
