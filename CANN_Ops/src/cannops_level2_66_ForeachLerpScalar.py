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

from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x1: List[torch.Tensor], x2: List[torch.Tensor], scalar_weight: float) -> List[torch.Tensor]:
        """
        Native PyTorch implementation of ForeachLerpScalar.
        Performs y_i = x1_i + weight * (x2_i - x1_i) for each tensor in the lists.
        """
        if not (isinstance(x1, list) and isinstance(x2, list)):
            raise TypeError('Inputs x1 and x2 must be lists of tensors.')
        if len(x1) != len(x2):
            raise ValueError('Input tensor lists x1 and x2 must have the same length.')
        output_list = []
        for i in range(len(x1)):
            result_tensor = torch.lerp(x1[i], x2[i], scalar_weight)
            output_list.append(result_tensor)
        return output_list

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level2_66_ForeachLerpScalar.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        x1_info = inputs[0]
        x2_info = inputs[1]
        scalar_weight_info = inputs[2]

        x1 = []
        for _shape in x1_info["shapes"]:
            _t = torch.rand({"dtype": x1_info["dtype"], "shape": _shape, "range": x1_info.get("range", [0, 1]), "mean": x1_info.get("mean", 0.0), "std": x1_info.get("std", 1.0), "value": x1_info.get("value")}["shape"], dtype=DTYPE_MAP[{"dtype": x1_info["dtype"], "shape": _shape, "range": x1_info.get("range", [0, 1]), "mean": x1_info.get("mean", 0.0), "std": x1_info.get("std", 1.0), "value": x1_info.get("value")}["dtype"]])
            x1.append(_t)
        x2 = []
        for _shape in x2_info["shapes"]:
            _t = torch.rand({"dtype": x2_info["dtype"], "shape": _shape, "range": x2_info.get("range", [0, 1]), "mean": x2_info.get("mean", 0.0), "std": x2_info.get("std", 1.0), "value": x2_info.get("value")}["shape"], dtype=DTYPE_MAP[{"dtype": x2_info["dtype"], "shape": _shape, "range": x2_info.get("range", [0, 1]), "mean": x2_info.get("mean", 0.0), "std": x2_info.get("std", 1.0), "value": x2_info.get("value")}["dtype"]])
            x2.append(_t)
        scalar_weight = scalar_weight_info["value"]

        input_groups.append([x1, x2, scalar_weight])
    return input_groups


def get_init_inputs():
    return []
