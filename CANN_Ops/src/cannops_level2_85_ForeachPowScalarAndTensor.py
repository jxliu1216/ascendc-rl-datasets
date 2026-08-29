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

from typing import List, Tuple
import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x: List[torch.Tensor], scalar_weight) -> List[torch.Tensor]:
        """
        """
        if not isinstance(x, list):
            raise TypeError('Inputs x must be lists of tensors.')
        output_list = []
        for i in range(len(x)):
            result_tensor = torch.pow(scalar_weight, x[i])
            output_list.append(result_tensor)
        return output_list

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level2_85_ForeachPowScalarAndTensor.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        x_info = inputs[0]
        scalar_weight_info = inputs[1]

        x = []
        for _shape in x_info["shapes"]:
            _t = torch.rand({"dtype": x_info["dtype"], "shape": _shape, "range": x_info.get("range", [0, 1]), "mean": x_info.get("mean", 0.0), "std": x_info.get("std", 1.0), "value": x_info.get("value")}["shape"], dtype=DTYPE_MAP[{"dtype": x_info["dtype"], "shape": _shape, "range": x_info.get("range", [0, 1]), "mean": x_info.get("mean", 0.0), "std": x_info.get("std", 1.0), "value": x_info.get("value")}["dtype"]])
            x.append(_t)
        scalar_weight = scalar_weight_info["value"]

        input_groups.append([x, scalar_weight])
    return input_groups


def get_init_inputs():
    return []
