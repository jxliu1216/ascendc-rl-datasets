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

    def __init__(self, n: int, incx: int):
        super(Model, self).__init__()
        self.n = n
        self.incx = incx

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.incx == 1:
            sliced_x = x[:self.n]
            original_indices_selected = torch.arange(0, self.n, device=x.device).long()
        else:
            end_val = float(self.n) * float(self.incx)
            step_val = float(self.incx)
            indices = torch.arange(0.0, end_val, step_val, device=x.device).long()
            sliced_x = x.index_select(0, indices)
            original_indices_selected = indices
        abs_real_imag = torch.abs(sliced_x.real) + torch.abs(sliced_x.imag)
        argmin_idx_in_sliced = torch.argmin(abs_real_imag)
        golden_0_based_idx = original_indices_selected[argmin_idx_in_sliced]
        golden = golden_0_based_idx + 1
        return golden.to(torch.int32)

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level1_30_Icamin.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        x_info = inputs[0]

        if "data" in x_info:
            x = torch.tensor(x_info["data"], dtype=DTYPE_MAP[x_info["dtype"]]).reshape(x_info["shape"])
        else:
            x_re = torch.randn(x_info["shape"], dtype=torch.float32)
            x_im = torch.randn(x_info["shape"], dtype=torch.float32)
            x = torch.complex(x_re, x_im).to(DTYPE_MAP[x_info["dtype"]])

        input_groups.append([x])
    return input_groups


def get_init_inputs():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level1_30_Icamin.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    init_groups = []
    for case in cases:
        entries = case.get("init_inputs", [])
        n_info = entries[0]
        incx_info = entries[1]
        n = n_info["value"]
        incx = incx_info["value"]
        init_groups.append([n, incx])
    return init_groups
