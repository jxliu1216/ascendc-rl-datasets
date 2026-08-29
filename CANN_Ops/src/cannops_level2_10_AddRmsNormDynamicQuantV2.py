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

def golden_add_rms_norm(x1: torch.Tensor, x2: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps: float) -> tuple:
    """与 ATK goldenAddRmsNorm 一致: x=x1+x2, rstd=rsqrt(mean(x^2)+eps), y=x*rstd*gamma+beta, return (y, x)."""
    ori_dtype = x1.dtype
    if ori_dtype != torch.float32:
        x1 = x1.float()
        x2 = x2.float()
        gamma = gamma.float()
        beta = beta.float()
    x = x1 + x2
    rstd = torch.rsqrt(x.pow(2).mean(axis=-1, keepdim=True) + eps)
    y = x * rstd * gamma + beta
    if ori_dtype != torch.float32:
        return (y, x.to(ori_dtype))
    return (y, x)

def golden_dynamic_quant(x: torch.Tensor, smooth: Optional[torch.Tensor]) -> tuple:
    """与 ATK goldenDynamicQuant 一致: smooth_x = x 或 x*smooth; gs_rev=127/max(|smooth_x|), gs=1/gs_rev, gq=round(smooth_x*gs_rev).int8."""
    x = x.float() if x.dtype != torch.float32 else x
    if smooth is not None:
        smooth = smooth.float() if smooth.dtype != torch.float32 else smooth
    else:
        smooth = None
    smooth_x = x if smooth is None else x * smooth
    x_max = torch.max(torch.abs(smooth_x), dim=-1, keepdim=True)[0].clamp(min=1e-08)
    gs_rev = 127.0 / x_max
    gs = 1.0 / gs_rev
    sx = smooth_x * gs_rev
    gq = torch.round(sx).to(torch.int8)
    return (gq, gs.squeeze(-1).float())

class Model(nn.Module):
    """Reference: add + RMSNorm(+beta) + dynamic quant，与 ATK FunctionRmsNormGradApi 标杆一致。"""

    def __init__(self, gamma: torch.Tensor, epsilon: float=1e-06, beta: Optional[torch.Tensor]=None):
        super(Model, self).__init__()
        self.gamma = gamma.detach().to(torch.float32)
        self.epsilon = epsilon
        self.beta = beta.detach().to(torch.float32) if beta is not None else None

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, smooth1: Optional[torch.Tensor]=None, smooth2: Optional[torch.Tensor]=None) -> List[torch.Tensor]:
        device = x1.device
        dtype = x1.dtype
        gamma = self.gamma.to(device).to(dtype)
        beta = self.beta.to(device).to(dtype) if self.beta is not None else torch.zeros(gamma.shape, device=device, dtype=dtype)
        gy_fp32, gx = golden_add_rms_norm(x1, x2, gamma, beta, self.epsilon)
        smooth1_exist = smooth1 is not None
        smooth2_exist = smooth2 is not None
        if smooth1_exist and smooth2_exist:
            gq1, gs1 = golden_dynamic_quant(gy_fp32, smooth1)
            gq2, gs2 = golden_dynamic_quant(gy_fp32, smooth2)
        elif smooth1_exist and (not smooth2_exist):
            gq1, gs1 = golden_dynamic_quant(gy_fp32, smooth1)
            gq2 = torch.zeros_like(gq1, device=device, dtype=gq1.dtype)
            gs2 = torch.zeros_like(gs1, device=device, dtype=gs1.dtype)
        elif not smooth1_exist and (not smooth2_exist):
            gq1, gs1 = golden_dynamic_quant(gy_fp32, None)
            gq2 = torch.zeros_like(gq1, device=device, dtype=gq1.dtype)
            gs2 = torch.zeros_like(gs1, device=device, dtype=gs1.dtype)
        else:
            gq1, gs1 = golden_dynamic_quant(gy_fp32, None)
            gq2, gs2 = golden_dynamic_quant(gy_fp32, smooth2)
        y3 = gy_fp32
        y4 = gy_fp32.to(dtype) if dtype != torch.float32 else gy_fp32
        x_out = gx
        N = gq1.shape[0]
        scale1_out = gs1 if gs1.dim() >= 1 else gs1.unsqueeze(0).expand(N)
        scale2_out = gs2 if gs2.dim() >= 1 else gs2.unsqueeze(0).expand(N)
        return [gq1, gq2, y3, y4, x_out, scale1_out, scale2_out]

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level2_10_AddRmsNormDynamicQuantV2.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        arg0_info = inputs[0]
        arg1_info = inputs[1]

        if "data" in arg0_info:
            arg0 = torch.tensor(arg0_info["data"], dtype=DTYPE_MAP[arg0_info["dtype"]]).reshape(arg0_info["shape"])
        else:
            arg0 = torch.rand(arg0_info["shape"], dtype=DTYPE_MAP[arg0_info["dtype"]]) * (arg0_info["range"][1] - arg0_info["range"][0]) + arg0_info["range"][0]
        if "data" in arg1_info:
            arg1 = torch.tensor(arg1_info["data"], dtype=DTYPE_MAP[arg1_info["dtype"]]).reshape(arg1_info["shape"])
        else:
            arg1 = torch.rand(arg1_info["shape"], dtype=DTYPE_MAP[arg1_info["dtype"]]) * (arg1_info["range"][1] - arg1_info["range"][0]) + arg1_info["range"][0]

        input_groups.append([arg0, arg1])
    return input_groups


def get_init_inputs():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level2_10_AddRmsNormDynamicQuantV2.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    init_groups = []
    for case in cases:
        entries = case.get("init_inputs", [])
        init0_info = entries[0]
        init1_info = entries[1]
        if "data" in init0_info:
            init0 = torch.tensor(init0_info["data"], dtype=DTYPE_MAP[init0_info["dtype"]]).reshape(init0_info["shape"])
        else:
            init0 = torch.rand(init0_info["shape"], dtype=DTYPE_MAP[init0_info["dtype"]])
        init1 = init1_info["value"]
        init_groups.append([init0, init1])
    return init_groups
