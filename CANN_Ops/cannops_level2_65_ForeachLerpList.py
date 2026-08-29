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
    Simple model that performs abs operation.
    """

    def __init__(self):
        """
        Initialize the model for abs operation.
        No parameters needed for basic abs operation.
        """
        super(Model, self).__init__()

    def forward(self, x1: List[torch.Tensor], x2: List[torch.Tensor], weight: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Computes the abs of input elements.

        Args:
            inputs: Input tensor list of any shape.

        Returns:
            Output tensor of same shape as input with abs applied elementwise.
        """
        x1_new = []
        x2_new = []
        weight_new = []
        for i in range(len(x1)):
            x1_new.append(x1[i].to(torch.float32))
            x2_new.append(x2[i].to(torch.float32))
            weight_new.append(weight[i].to(torch.float32))
        res = torch._foreach_lerp(x1_new, x2_new, weight_new)
        return [r.to(x1[0].dtype) for r in res]

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level2_65_ForeachLerpList.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        x1_info = inputs[0]
        x2_info = inputs[1]
        weight_info = inputs[2]

        x1 = []
        for _shape in x1_info["shapes"]:
            _t = torch.rand({"dtype": x1_info["dtype"], "shape": _shape, "range": x1_info.get("range", [0, 1]), "mean": x1_info.get("mean", 0.0), "std": x1_info.get("std", 1.0), "value": x1_info.get("value")}["shape"], dtype=DTYPE_MAP[{"dtype": x1_info["dtype"], "shape": _shape, "range": x1_info.get("range", [0, 1]), "mean": x1_info.get("mean", 0.0), "std": x1_info.get("std", 1.0), "value": x1_info.get("value")}["dtype"]]) * ({"dtype": x1_info["dtype"], "shape": _shape, "range": x1_info.get("range", [0, 1]), "mean": x1_info.get("mean", 0.0), "std": x1_info.get("std", 1.0), "value": x1_info.get("value")}["range"][1] - {"dtype": x1_info["dtype"], "shape": _shape, "range": x1_info.get("range", [0, 1]), "mean": x1_info.get("mean", 0.0), "std": x1_info.get("std", 1.0), "value": x1_info.get("value")}["range"][0]) + {"dtype": x1_info["dtype"], "shape": _shape, "range": x1_info.get("range", [0, 1]), "mean": x1_info.get("mean", 0.0), "std": x1_info.get("std", 1.0), "value": x1_info.get("value")}["range"][0]
            x1.append(_t)
        x2 = []
        for _shape in x2_info["shapes"]:
            _t = torch.rand({"dtype": x2_info["dtype"], "shape": _shape, "range": x2_info.get("range", [0, 1]), "mean": x2_info.get("mean", 0.0), "std": x2_info.get("std", 1.0), "value": x2_info.get("value")}["shape"], dtype=DTYPE_MAP[{"dtype": x2_info["dtype"], "shape": _shape, "range": x2_info.get("range", [0, 1]), "mean": x2_info.get("mean", 0.0), "std": x2_info.get("std", 1.0), "value": x2_info.get("value")}["dtype"]]) * ({"dtype": x2_info["dtype"], "shape": _shape, "range": x2_info.get("range", [0, 1]), "mean": x2_info.get("mean", 0.0), "std": x2_info.get("std", 1.0), "value": x2_info.get("value")}["range"][1] - {"dtype": x2_info["dtype"], "shape": _shape, "range": x2_info.get("range", [0, 1]), "mean": x2_info.get("mean", 0.0), "std": x2_info.get("std", 1.0), "value": x2_info.get("value")}["range"][0]) + {"dtype": x2_info["dtype"], "shape": _shape, "range": x2_info.get("range", [0, 1]), "mean": x2_info.get("mean", 0.0), "std": x2_info.get("std", 1.0), "value": x2_info.get("value")}["range"][0]
            x2.append(_t)
        weight = []
        for _shape in weight_info["shapes"]:
            _t = torch.rand({"dtype": weight_info["dtype"], "shape": _shape, "range": weight_info.get("range", [0, 1]), "mean": weight_info.get("mean", 0.0), "std": weight_info.get("std", 1.0), "value": weight_info.get("value")}["shape"], dtype=DTYPE_MAP[{"dtype": weight_info["dtype"], "shape": _shape, "range": weight_info.get("range", [0, 1]), "mean": weight_info.get("mean", 0.0), "std": weight_info.get("std", 1.0), "value": weight_info.get("value")}["dtype"]]) * ({"dtype": weight_info["dtype"], "shape": _shape, "range": weight_info.get("range", [0, 1]), "mean": weight_info.get("mean", 0.0), "std": weight_info.get("std", 1.0), "value": weight_info.get("value")}["range"][1] - {"dtype": weight_info["dtype"], "shape": _shape, "range": weight_info.get("range", [0, 1]), "mean": weight_info.get("mean", 0.0), "std": weight_info.get("std", 1.0), "value": weight_info.get("value")}["range"][0]) + {"dtype": weight_info["dtype"], "shape": _shape, "range": weight_info.get("range", [0, 1]), "mean": weight_info.get("mean", 0.0), "std": weight_info.get("std", 1.0), "value": weight_info.get("value")}["range"][0]
            weight.append(_t)

        input_groups.append([x1, x2, weight])
    return input_groups


def get_init_inputs():
    return []
