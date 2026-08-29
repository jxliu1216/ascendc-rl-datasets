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

    def forward(self, start: torch.Tensor, end: torch.Tensor, step: torch.Tensor) -> torch.Tensor:
        return torch.arange(start.item(), end.item(), step.item())

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level1_7_Arange.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        start_info = inputs[0]
        end_info = inputs[1]
        step_info = inputs[2]

        if "data" in start_info:
            start = torch.tensor(start_info["data"], dtype=DTYPE_MAP[start_info["dtype"]]).reshape(start_info["shape"])
        else:
            _dt = DTYPE_MAP[start_info["dtype"]]
            if _dt in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
                start = torch.randint(start_info["range"][0], start_info["range"][1] + 1, tuple(start_info["shape"]), dtype=_dt)
            elif _dt == torch.bool:
                start = torch.rand(start_info["shape"]) > 0.5
            else:
                start = torch.rand(start_info["shape"], dtype=_dt) * (start_info["range"][1] - start_info["range"][0]) + start_info["range"][0]
        if "data" in end_info:
            end = torch.tensor(end_info["data"], dtype=DTYPE_MAP[end_info["dtype"]]).reshape(end_info["shape"])
        else:
            _dt = DTYPE_MAP[end_info["dtype"]]
            if _dt in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
                end = torch.randint(end_info["range"][0], end_info["range"][1] + 1, tuple(end_info["shape"]), dtype=_dt)
            elif _dt == torch.bool:
                end = torch.rand(end_info["shape"]) > 0.5
            else:
                end = torch.rand(end_info["shape"], dtype=_dt) * (end_info["range"][1] - end_info["range"][0]) + end_info["range"][0]
        if "data" in step_info:
            step = torch.tensor(step_info["data"], dtype=DTYPE_MAP[step_info["dtype"]]).reshape(step_info["shape"])
        else:
            _dt = DTYPE_MAP[step_info["dtype"]]
            if _dt in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
                step = torch.randint(step_info["range"][0], step_info["range"][1] + 1, tuple(step_info["shape"]), dtype=_dt)
            elif _dt == torch.bool:
                step = torch.rand(step_info["shape"]) > 0.5
            else:
                step = torch.rand(step_info["shape"], dtype=_dt) * (step_info["range"][1] - step_info["range"][0]) + step_info["range"][0]

        input_groups.append([start, end, step])
    return input_groups


def get_init_inputs():
    return []
