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

    def forward(self, input: torch.tensor, num_rows: int, num_columns: int=0, batch_shape: list[int]=[], dtype: int=0) -> torch.Tensor:
        eye_matrix = torch.eye(num_rows, num_columns).to(input.device).to(input.dtype)
        res_shape = batch_shape + [num_rows, num_columns]
        res = eye_matrix.expand(*res_shape)
        return res

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level1_20_Eye.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        input_info = inputs[0]
        num_rows_info = inputs[1]
        num_columns_info = inputs[2]
        batch_shape_info = inputs[3]
        dtype_info = inputs[4]

        if "data" in input_info:
            input = torch.tensor(input_info["data"], dtype=DTYPE_MAP[input_info["dtype"]]).reshape(input_info["shape"])
        else:
            _dt = DTYPE_MAP[input_info["dtype"]]
            if _dt in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
                input = torch.full(input_info["shape"], input_info["fill"], dtype=_dt)
            elif _dt == torch.bool:
                input = torch.rand(input_info["shape"]) > 0.5
            else:
                input = torch.full(input_info["shape"], input_info["fill"], dtype=_dt)
        num_rows = num_rows_info["value"]
        num_columns = num_columns_info["value"]
        batch_shape = batch_shape_info["value"]
        dtype = dtype_info["value"]

        input_groups.append([input, num_rows, num_columns, batch_shape, dtype])
    return input_groups


def get_init_inputs():
    return []
