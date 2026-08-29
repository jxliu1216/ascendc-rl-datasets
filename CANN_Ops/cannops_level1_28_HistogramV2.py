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

    def forward(self, input: torch.Tensor, min: torch.Tensor, max: torch.Tensor, bins: int) -> torch.Tensor:
        return torch.histc(input.to(torch.float32), bins=bins, min=min[0].item(), max=max[0].item()).to(torch.int32)

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level1_28_HistogramV2.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        input_info = inputs[0]
        min_info = inputs[1]
        max_info = inputs[2]
        bins_info = inputs[3]

        if "data" in input_info:
            input = torch.tensor(input_info["data"], dtype=DTYPE_MAP[input_info["dtype"]]).reshape(input_info["shape"])
        else:
            _dt = DTYPE_MAP[input_info["dtype"]]
            if _dt in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
                input = torch.randint(input_info["range"][0], input_info["range"][1] + 1, tuple(input_info["shape"]), dtype=_dt)
            elif _dt == torch.bool:
                input = torch.rand(input_info["shape"]) > 0.5
            else:
                input = torch.randn(input_info["shape"], dtype=_dt)
        if "data" in min_info:
            min = torch.tensor(min_info["data"], dtype=DTYPE_MAP[min_info["dtype"]]).reshape(min_info["shape"])
        else:
            _dt = DTYPE_MAP[min_info["dtype"]]
            if _dt in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
                min = torch.randint(min_info["range"][0], min_info["range"][1] + 1, tuple(min_info["shape"]), dtype=_dt)
            elif _dt == torch.bool:
                min = torch.rand(min_info["shape"]) > 0.5
            else:
                min = torch.rand(min_info["shape"], dtype=_dt) * (min_info["range"][1] - min_info["range"][0]) + min_info["range"][0]
        if "data" in max_info:
            max = torch.tensor(max_info["data"], dtype=DTYPE_MAP[max_info["dtype"]]).reshape(max_info["shape"])
        else:
            _dt = DTYPE_MAP[max_info["dtype"]]
            if _dt in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
                max = torch.randint(max_info["range"][0], max_info["range"][1] + 1, tuple(max_info["shape"]), dtype=_dt)
            elif _dt == torch.bool:
                max = torch.rand(max_info["shape"]) > 0.5
            else:
                max = torch.rand(max_info["shape"], dtype=_dt) * (max_info["range"][1] - max_info["range"][0]) + max_info["range"][0]
        bins = bins_info["value"]

        input_groups.append([input, min, max, bins])
    return input_groups


def get_init_inputs():
    return []
