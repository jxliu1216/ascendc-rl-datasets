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

    def forward(self, grad_output: torch.Tensor, output_size: List[int], input_size: List[int], align_corners: bool=False, scale_d: float=0.0, scale_h: float=0.0, scale_w: float=0.0) -> torch.Tensor:
        N, C, oD, oH, oW = grad_output.shape
        _, _, iD, iH, iW = input_size
        if align_corners:
            scale_d_val = (iD - 1) / (oD - 1) if oD > 1 else 0.0
            scale_h_val = (iH - 1) / (oH - 1) if oH > 1 else 0.0
            scale_w_val = (iW - 1) / (oW - 1) if oW > 1 else 0.0
        else:
            if scale_d > 0.0:
                scale_d_val = scale_d
            else:
                scale_d_val = float(iD) / oD if oD != 0 else 0.0
            if scale_h > 0.0:
                scale_h_val = scale_h
            else:
                scale_h_val = float(iH) / oH if oH != 0 else 0.0
            if scale_w > 0.0:
                scale_w_val = scale_w
            else:
                scale_w_val = float(iW) / oW if oW != 0 else 0.0
        grad_input = torch.zeros(N, C, iD, iH, iW, device=grad_output.device, dtype=grad_output.dtype)
        grad_out_f = grad_output.float()
        grad_in_f = torch.zeros(N, C, iD, iH, iW, device=grad_output.device, dtype=torch.float32)
        for d_out in range(oD):
            for h_out in range(oH):
                for w_out in range(oW):
                    if align_corners:
                        d_src = scale_d_val * d_out
                        h_src = scale_h_val * h_out
                        w_src = scale_w_val * w_out
                    else:
                        d_src = max(scale_d_val * (d_out + 0.5) - 0.5, 0.0)
                        h_src = max(scale_h_val * (h_out + 0.5) - 0.5, 0.0)
                        w_src = max(scale_w_val * (w_out + 0.5) - 0.5, 0.0)
                    d0 = int(d_src)
                    h0 = int(h_src)
                    w0 = int(w_src)
                    d1 = min(d0 + 1, iD - 1)
                    h1 = min(h0 + 1, iH - 1)
                    w1 = min(w0 + 1, iW - 1)
                    d0 = min(d0, iD - 1)
                    h0 = min(h0, iH - 1)
                    w0 = min(w0, iW - 1)
                    ld = d_src - d0 if d0 != d1 else 0.0
                    lh = h_src - h0 if h0 != h1 else 0.0
                    lw = w_src - w0 if w0 != w1 else 0.0
                    w000 = (1 - ld) * (1 - lh) * (1 - lw)
                    w001 = (1 - ld) * (1 - lh) * lw
                    w010 = (1 - ld) * lh * (1 - lw)
                    w011 = (1 - ld) * lh * lw
                    w100 = ld * (1 - lh) * (1 - lw)
                    w101 = ld * (1 - lh) * lw
                    w110 = ld * lh * (1 - lw)
                    w111 = ld * lh * lw
                    g = grad_out_f[:, :, d_out, h_out, w_out]
                    grad_in_f[:, :, d0, h0, w0] += g * w000
                    grad_in_f[:, :, d0, h0, w1] += g * w001
                    grad_in_f[:, :, d0, h1, w0] += g * w010
                    grad_in_f[:, :, d0, h1, w1] += g * w011
                    grad_in_f[:, :, d1, h0, w0] += g * w100
                    grad_in_f[:, :, d1, h0, w1] += g * w101
                    grad_in_f[:, :, d1, h1, w0] += g * w110
                    grad_in_f[:, :, d1, h1, w1] += g * w111
        return grad_in_f

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level2_168_UpsampleTrilinear3dBackward.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        arg0_info = inputs[0]
        arg1_info = inputs[1]
        arg2_info = inputs[2]
        arg3_info = inputs[3]

        if "data" in arg0_info:
            arg0 = torch.tensor(arg0_info["data"], dtype=DTYPE_MAP[arg0_info["dtype"]]).reshape(arg0_info["shape"])
        else:
            arg0 = torch.randn(arg0_info["shape"], dtype=DTYPE_MAP[arg0_info["dtype"]])
        arg1 = arg1_info["value"]
        arg2 = arg2_info["value"]
        arg3 = arg3_info["value"]

        input_groups.append([arg0, arg1, arg2, arg3])
    return input_groups


def get_init_inputs():
    return []
