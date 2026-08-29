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

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, var, indices, updates, axis=-1, reduction='none'):
        result = var.clone()
        if reduction == 'none':
            result.scatter_(axis, indices.to(torch.int64), updates)
        elif reduction == 'add':
            result.scatter_add_(axis, indices.to(torch.int64), updates)
        return result

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level1_59_ScatterElementsV2.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        var_info = inputs[0]
        indices_info = inputs[1]
        updates_info = inputs[2]
        axis_info = inputs[3]
        reduction_info = inputs[4]

        if "data" in var_info:
            var = torch.tensor(var_info["data"], dtype=DTYPE_MAP[var_info["dtype"]]).reshape(var_info["shape"])
        else:
            _dt = DTYPE_MAP[var_info["dtype"]]
            if _dt in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
                var = torch.randint(var_info["range"][0], var_info["range"][1] + 1, tuple(var_info["shape"]), dtype=_dt)
            elif _dt == torch.bool:
                var = torch.rand(var_info["shape"]) > 0.5
            else:
                var = torch.randn(var_info["shape"], dtype=_dt) * var_info["std"] + var_info["mean"]
        if "data" in indices_info:
            indices = torch.tensor(indices_info["data"], dtype=DTYPE_MAP[indices_info["dtype"]]).reshape(indices_info["shape"])
        else:
            indices = torch.randint(indices_info["range"][0], indices_info["range"][1] + 1, tuple(indices_info["shape"]), dtype=DTYPE_MAP[indices_info["dtype"]])
        if "data" in updates_info:
            updates = torch.tensor(updates_info["data"], dtype=DTYPE_MAP[updates_info["dtype"]]).reshape(updates_info["shape"])
        else:
            _dt = DTYPE_MAP[updates_info["dtype"]]
            if _dt in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
                updates = torch.randint(updates_info["range"][0], updates_info["range"][1] + 1, tuple(updates_info["shape"]), dtype=_dt)
            elif _dt == torch.bool:
                updates = torch.rand(updates_info["shape"]) > 0.5
            else:
                updates = torch.randn(updates_info["shape"], dtype=_dt) * updates_info["std"] + updates_info["mean"]
        axis = axis_info["value"]
        reduction = reduction_info["value"]

        input_groups.append([var, indices, updates, axis, reduction])
    return input_groups


def get_init_inputs():
    return []
