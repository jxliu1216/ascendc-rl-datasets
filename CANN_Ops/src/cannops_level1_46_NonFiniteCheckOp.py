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
from typing import List

class Model(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, xs: list[torch.Tensor]) -> torch.Tensor:
        flags = [float(torch.isnan(x).any() or torch.isinf(x).any()) for x in xs]
        return torch.tensor(sum(flags) > 0, dtype=torch.float32)

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level1_46_NonFiniteCheckOp.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        xs_info = inputs[0]

        xs = []
        for _shape in xs_info["shapes"]:
            _t = torch.randn({"dtype": xs_info["dtype"], "shape": _shape, "range": xs_info.get("range", [0, 1]), "mean": xs_info.get("mean", 0.0), "std": xs_info.get("std", 1.0), "value": xs_info.get("value")}["shape"], dtype=DTYPE_MAP[{"dtype": xs_info["dtype"], "shape": _shape, "range": xs_info.get("range", [0, 1]), "mean": xs_info.get("mean", 0.0), "std": xs_info.get("std", 1.0), "value": xs_info.get("value")}["dtype"]]) * {"dtype": xs_info["dtype"], "shape": _shape, "range": xs_info.get("range", [0, 1]), "mean": xs_info.get("mean", 0.0), "std": xs_info.get("std", 1.0), "value": xs_info.get("value")}["std"] + {"dtype": xs_info["dtype"], "shape": _shape, "range": xs_info.get("range", [0, 1]), "mean": xs_info.get("mean", 0.0), "std": xs_info.get("std", 1.0), "value": xs_info.get("value")}["mean"]
            xs.append(_t)
        if xs_info.get("inject"):
            _f = xs[0].reshape(-1)
            _f[0] = float(xs_info["inject"])
            xs[0] = _f.reshape(xs[0].shape)

        input_groups.append([xs])
    return input_groups


def get_init_inputs():
    return []
