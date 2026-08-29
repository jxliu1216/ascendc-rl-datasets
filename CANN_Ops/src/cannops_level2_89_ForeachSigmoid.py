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
        super(Model, self).__init__()

    def forward(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        return [torch.sigmoid(x) for x in inputs]

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level2_89_ForeachSigmoid.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        inputs_info = inputs[0]

        inputs = []
        for _shape in inputs_info["shapes"]:
            _t = torch.rand({"dtype": inputs_info["dtype"], "shape": _shape, "range": inputs_info.get("range", [0, 1]), "mean": inputs_info.get("mean", 0.0), "std": inputs_info.get("std", 1.0), "value": inputs_info.get("value")}["shape"], dtype=DTYPE_MAP[{"dtype": inputs_info["dtype"], "shape": _shape, "range": inputs_info.get("range", [0, 1]), "mean": inputs_info.get("mean", 0.0), "std": inputs_info.get("std", 1.0), "value": inputs_info.get("value")}["dtype"]]) * ({"dtype": inputs_info["dtype"], "shape": _shape, "range": inputs_info.get("range", [0, 1]), "mean": inputs_info.get("mean", 0.0), "std": inputs_info.get("std", 1.0), "value": inputs_info.get("value")}["range"][1] - {"dtype": inputs_info["dtype"], "shape": _shape, "range": inputs_info.get("range", [0, 1]), "mean": inputs_info.get("mean", 0.0), "std": inputs_info.get("std", 1.0), "value": inputs_info.get("value")}["range"][0]) + {"dtype": inputs_info["dtype"], "shape": _shape, "range": inputs_info.get("range", [0, 1]), "mean": inputs_info.get("mean", 0.0), "std": inputs_info.get("std", 1.0), "value": inputs_info.get("value")}["range"][0]
            inputs.append(_t)

        input_groups.append([inputs])
    return input_groups


def get_init_inputs():
    return []
