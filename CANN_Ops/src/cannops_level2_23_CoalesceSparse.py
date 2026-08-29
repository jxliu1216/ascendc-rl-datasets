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

from typing import Tuple, Optional
import torch
import torch.nn as nn
from typing import List
import torch.nn.functional as F

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, indices: torch.Tensor, values: torch.Tensor, sparse_tensor: torch.Tensor):
        sparse_tensor = sparse_tensor.coalesce()
        new_indices = sparse_tensor.indices()
        new_values = sparse_tensor.values()
        new_indices = torch.transpose(new_indices, 0, 1)
        return [new_indices, new_values]

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level2_23_CoalesceSparse.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        indices_info = inputs[0]
        values_info = inputs[1]
        sparse_tensor_info = inputs[2]

        if "data" in indices_info:
            indices = torch.tensor(indices_info["data"], dtype=DTYPE_MAP[indices_info["dtype"]]).reshape(indices_info["shape"])
        else:
            indices = torch.randint(indices_info["range"][0], indices_info["range"][1] + 1, tuple(indices_info["shape"]), dtype=DTYPE_MAP[indices_info["dtype"]])
        if "data" in values_info:
            values = torch.tensor(values_info["data"], dtype=DTYPE_MAP[values_info["dtype"]]).reshape(values_info["shape"])
        else:
            _dt = DTYPE_MAP[values_info["dtype"]]
            if _dt in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
                values = torch.randint(values_info["range"][0], values_info["range"][1] + 1, tuple(values_info["shape"]), dtype=_dt)
            elif _dt == torch.bool:
                values = torch.rand(values_info["shape"]) > 0.5
            else:
                values = torch.randn(values_info["shape"], dtype=_dt)
        sparse_tensor = torch.sparse_coo_tensor(indices, values, sparse_tensor_info["shape"])

        input_groups.append([indices, values, sparse_tensor])
    return input_groups


def get_init_inputs():
    return []
