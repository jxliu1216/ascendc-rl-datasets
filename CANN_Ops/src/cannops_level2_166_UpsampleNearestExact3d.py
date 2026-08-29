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

    def forward(self, x: torch.Tensor, output_size: List[int], scale_d: float=0.0, scale_h: float=0.0, scale_w: float=0.0) -> torch.Tensor:
        N, C, iD, iH, iW = x.shape
        oD, oH, oW = output_size
        s_d = iD / oD if scale_d <= 0.0 else scale_d
        s_h = iH / oH if scale_h <= 0.0 else scale_h
        s_w = iW / oW if scale_w <= 0.0 else scale_w
        d_idx = torch.arange(oD, device=x.device, dtype=torch.float32)
        h_idx = torch.arange(oH, device=x.device, dtype=torch.float32)
        w_idx = torch.arange(oW, device=x.device, dtype=torch.float32)
        d_src = ((d_idx + 0.5) * s_d).floor().long().clamp(0, iD - 1)
        h_src = ((h_idx + 0.5) * s_h).floor().long().clamp(0, iH - 1)
        w_src = ((w_idx + 0.5) * s_w).floor().long().clamp(0, iW - 1)
        return x[:, :, d_src.unsqueeze(1).unsqueeze(1), h_src.unsqueeze(1), w_src]

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level2_166_UpsampleNearestExact3d.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        x_info = inputs[0]
        output_size_info = inputs[1]
        scale_d_info = inputs[2]
        scale_h_info = inputs[3]
        scale_w_info = inputs[4]

        if "data" in x_info:
            x = torch.tensor(x_info["data"], dtype=DTYPE_MAP[x_info["dtype"]]).reshape(x_info["shape"])
        else:
            x = torch.randn(x_info["shape"], dtype=DTYPE_MAP[x_info["dtype"]])
        output_size = output_size_info["value"]
        scale_d = scale_d_info["value"]
        scale_h = scale_h_info["value"]
        scale_w = scale_w_info["value"]

        input_groups.append([x, output_size, scale_d, scale_h, scale_w])
    return input_groups


def get_init_inputs():
    return []
