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

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input: torch.Tensor, n: Optional[int], norm: int) -> torch.Tensor:
        if norm == 1:
            norm_str = 'backward'
        elif norm == 2:
            norm_str = 'forward'
        elif norm == 3:
            norm_str = 'ortho'
        else:
            norm_str = 'backward'
        x = input.to(torch.float32)
        dim = -1
        output = torch.fft.rfft(x, n, dim, norm_str)
        output = torch.stack([output.real, output.imag], dim=-1)
        return output.flatten(-2)

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level1_55_Rfft1D.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        input_info = inputs[0]
        n_info = inputs[1]
        norm_info = inputs[2]

        if "data" in input_info:
            input = torch.tensor(input_info["data"], dtype=DTYPE_MAP[input_info["dtype"]]).reshape(input_info["shape"])
        else:
            input = torch.randn(input_info["shape"], dtype=DTYPE_MAP[input_info["dtype"]])
        if n_info["type"] == "attr":
            if n_info.get("dtype") == "none":
                n = None
            else:
                n = n_info["value"]
        norm = norm_info["value"]

        input_groups.append([input, n, norm])
    return input_groups


def get_init_inputs():
    return []
