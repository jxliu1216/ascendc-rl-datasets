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

def _dynamic_quant_per_row(y: torch.Tensor, smooth: Optional[torch.Tensor]=None, dim: int=-1) -> tuple:
    """Per-row dynamic quant. If smooth is not None: quantize y*smooth (smooth broadcast to y); else quantize y.
    scale = max(abs(smooth_x))/127, q = round(smooth_x/scale). Aligned with ATK goldenDynamicQuant."""
    y_flat = y.float()
    if smooth is not None:
        smooth_f = smooth.float().to(y_flat.device)
        smooth_x = y_flat * smooth_f
    else:
        smooth_x = y_flat
    scale = smooth_x.abs().amax(dim=dim, keepdim=True).clamp(min=1e-08) / 127.0
    q = (smooth_x / scale).round().clamp(-128, 127).to(torch.int8)
    return (q, scale.squeeze(dim).float())

def _golden_add_rms_norm(x1: torch.Tensor, x2: torch.Tensor, gamma: torch.Tensor, eps: float):
    """x = x1+x2, rstd = rsqrt(mean(x^2,-1)+eps), y = x * rstd * gamma. Aligned with ATK goldenAddRmsNorm (no beta)."""
    ori_dtype = x1.dtype
    x1_f = x1.float()
    x2_f = x2.float()
    gamma_f = gamma.float().to(x1_f.device)
    x = x1_f + x2_f
    rstd = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    y = x * rstd * gamma_f
    x_out = x if ori_dtype == torch.float32 else x.to(ori_dtype)
    return (y, x_out)

class Model(nn.Module):
    """Reference: add + RMSNorm + dynamic quant. Supports optional smooth1/smooth2. Outputs: y1, y2, x, scale1, scale2.
    Aligned with ATK goldenAddRmsNorm + goldenDynamicQuant."""

    def __init__(self, gamma: torch.Tensor, epsilon: float=1e-06, smooth_scale1: Optional[torch.Tensor]=None, smooth_scale2: Optional[torch.Tensor]=None):
        super(Model, self).__init__()
        self.gamma = gamma.detach().to(torch.float32)
        self.epsilon = epsilon
        self.smooth_scale1 = smooth_scale1
        self.smooth_scale2 = smooth_scale2

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> List[torch.Tensor]:
        gamma = self.gamma.to(x1.device).to(x1.dtype)
        y, x_out = _golden_add_rms_norm(x1, x2, gamma, self.epsilon)
        smooth1 = self.smooth_scale1.to(y.device).to(y.dtype) if self.smooth_scale1 is not None else None
        smooth2 = self.smooth_scale2.to(y.device).to(y.dtype) if self.smooth_scale2 is not None else None
        if smooth1 is not None and smooth2 is not None:
            y1, scale1 = _dynamic_quant_per_row(y, smooth1, dim=-1)
            y2, scale2 = _dynamic_quant_per_row(y, smooth2, dim=-1)
        elif smooth1 is not None:
            y1, scale1 = _dynamic_quant_per_row(y, smooth1, dim=-1)
            y2 = torch.zeros_like(y1, device=y.device, dtype=torch.int8)
            scale2 = torch.zeros_like(scale1, device=y.device, dtype=torch.float32)
        else:
            y1, scale1 = _dynamic_quant_per_row(y, None, dim=-1)
            y2 = torch.zeros_like(y1, device=y.device, dtype=torch.int8)
            scale2 = torch.zeros_like(scale1, device=y.device, dtype=torch.float32)
        scale1_out = scale1 if scale1.dim() >= 1 else scale1.unsqueeze(0)
        return [y1, y2, x_out, scale1_out, scale2]

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level2_9_AddRmsNormDynamicQuant.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        x1_info = inputs[0]
        x2_info = inputs[1]

        if "data" in x1_info:
            x1 = torch.tensor(x1_info["data"], dtype=DTYPE_MAP[x1_info["dtype"]]).reshape(x1_info["shape"])
        else:
            x1 = torch.rand(x1_info["shape"], dtype=DTYPE_MAP[x1_info["dtype"]]) * (x1_info["range"][1] - x1_info["range"][0]) + x1_info["range"][0]
        if "data" in x2_info:
            x2 = torch.tensor(x2_info["data"], dtype=DTYPE_MAP[x2_info["dtype"]]).reshape(x2_info["shape"])
        else:
            x2 = torch.rand(x2_info["shape"], dtype=DTYPE_MAP[x2_info["dtype"]]) * (x2_info["range"][1] - x2_info["range"][0]) + x2_info["range"][0]

        input_groups.append([x1, x2])
    return input_groups


def get_init_inputs():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level2_9_AddRmsNormDynamicQuant.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    init_groups = []
    for case in cases:
        entries = case.get("init_inputs", [])
        gamma_info = entries[0]
        epsilon_info = entries[1]
        smooth_scale1_info = entries[2]
        smooth_scale2_info = entries[3]
        if "data" in gamma_info:
            gamma = torch.tensor(gamma_info["data"], dtype=DTYPE_MAP[gamma_info["dtype"]]).reshape(gamma_info["shape"])
        else:
            gamma = torch.rand(gamma_info["shape"], dtype=DTYPE_MAP[gamma_info["dtype"]])
        epsilon = epsilon_info["value"]
        if smooth_scale1_info["type"] == "attr":
            if smooth_scale1_info.get("dtype") == "none":
                smooth_scale1 = None
            else:
                smooth_scale1 = smooth_scale1_info["value"]
        else:
            if "data" in smooth_scale1_info:
                smooth_scale1 = torch.tensor(smooth_scale1_info["data"], dtype=DTYPE_MAP[smooth_scale1_info["dtype"]]).reshape(smooth_scale1_info["shape"])
            else:
                smooth_scale1 = torch.rand(smooth_scale1_info["shape"], dtype=DTYPE_MAP[smooth_scale1_info["dtype"]])
        if smooth_scale2_info["type"] == "attr":
            if smooth_scale2_info.get("dtype") == "none":
                smooth_scale2 = None
            else:
                smooth_scale2 = smooth_scale2_info["value"]
        else:
            if "data" in smooth_scale2_info:
                smooth_scale2 = torch.tensor(smooth_scale2_info["data"], dtype=DTYPE_MAP[smooth_scale2_info["dtype"]]).reshape(smooth_scale2_info["shape"])
            else:
                smooth_scale2 = torch.rand(smooth_scale2_info["shape"], dtype=DTYPE_MAP[smooth_scale2_info["dtype"]])
        init_groups.append([gamma, epsilon, smooth_scale1, smooth_scale2])
    return init_groups
