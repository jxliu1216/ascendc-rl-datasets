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
import torch.nn.functional as F

class Model(nn.Module):
    """PyTorch native reference implementation (golden model)."""

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, grad_output: torch.Tensor, output_size: List[int], input_size: List[int], scales_h: float=0.0, scales_w: float=0.0) -> torch.Tensor:
        orig_dtype = grad_output.dtype
        N, C, oH, oW = grad_output.shape
        _, _, iH, iW = input_size
        scale_h = iH / oH
        scale_w = iW / oW
        h_idx = torch.arange(oH, device=grad_output.device, dtype=torch.float32)
        w_idx = torch.arange(oW, device=grad_output.device, dtype=torch.float32)
        h_src = ((h_idx + 0.5) * scale_h).floor().long().clamp(0, iH - 1)
        w_src = ((w_idx + 0.5) * scale_w).floor().long().clamp(0, iW - 1)
        src_2d = h_src.unsqueeze(1) * iW + w_src.unsqueeze(0)
        src_flat = src_2d.reshape(-1)
        grad_out_flat = grad_output.reshape(N, C, oH * oW).float()
        src_expanded = src_flat.unsqueeze(0).unsqueeze(0).expand(N, C, -1)
        grad_input_flat = torch.zeros(N, C, iH * iW, device=grad_output.device, dtype=torch.float32)
        grad_input_flat.scatter_add_(2, src_expanded, grad_out_flat)
        result = grad_input_flat.reshape(N, C, iH, iW)
        if orig_dtype != torch.float32:
            result = result.to(orig_dtype)
        return result

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level2_165_UpsampleNearestExact2dGrad.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        arg0_info = inputs[0]
        arg1_info = inputs[1]
        arg2_info = inputs[2]

        if "data" in arg0_info:
            arg0 = torch.tensor(arg0_info["data"], dtype=DTYPE_MAP[arg0_info["dtype"]]).reshape(arg0_info["shape"])
        else:
            arg0 = torch.randn(arg0_info["shape"], dtype=DTYPE_MAP[arg0_info["dtype"]])
        arg1 = arg1_info["value"]
        arg2 = arg2_info["value"]

        input_groups.append([arg0, arg1, arg2])
    return input_groups


def get_init_inputs():
    return []
