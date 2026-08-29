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

from typing import List, Optional
import torch
import torch.nn as nn
import math

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x1: List[torch.Tensor], x2: List[torch.Tensor], n: int) -> torch.Tensor:
        out_shape = [x1[0].shape[0], x1[0].shape[1], x2[0].shape[2]]
        if x1[0].dtype != torch.float16:
            out = torch.zeros(out_shape).to(torch.float32)
            for i in range(n):
                out = out + torch.matmul(x1[i].to(torch.float32), x2[i].to(torch.float32))
            return out.to(x1[0].dtype)
        else:
            out = torch.zeros(out_shape).to(x1[0].dtype)
            for i in range(n):
                out = out + torch.matmul(x1[i], x2[i])
            return out

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level1_43_MulAddn.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        x1_info = inputs[0]
        x2_info = inputs[1]
        n_info = inputs[2]

        x1 = []
        for _shape in x1_info["shapes"]:
            _t = torch.randn({"dtype": x1_info["dtype"], "shape": _shape, "range": x1_info.get("range", [0, 1]), "mean": x1_info.get("mean", 0.0), "std": x1_info.get("std", 1.0), "value": x1_info.get("value")}["shape"], dtype=DTYPE_MAP[{"dtype": x1_info["dtype"], "shape": _shape, "range": x1_info.get("range", [0, 1]), "mean": x1_info.get("mean", 0.0), "std": x1_info.get("std", 1.0), "value": x1_info.get("value")}["dtype"]])
            x1.append(_t)
        x2 = []
        for _shape in x2_info["shapes"]:
            _t = torch.randn({"dtype": x2_info["dtype"], "shape": _shape, "range": x2_info.get("range", [0, 1]), "mean": x2_info.get("mean", 0.0), "std": x2_info.get("std", 1.0), "value": x2_info.get("value")}["shape"], dtype=DTYPE_MAP[{"dtype": x2_info["dtype"], "shape": _shape, "range": x2_info.get("range", [0, 1]), "mean": x2_info.get("mean", 0.0), "std": x2_info.get("std", 1.0), "value": x2_info.get("value")}["dtype"]])
            x2.append(_t)
        n = n_info["value"]

        input_groups.append([x1, x2, n])
    return input_groups


def get_init_inputs():
    return []
