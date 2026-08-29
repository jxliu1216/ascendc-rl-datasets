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
AXIS_MUL_MODE = -65535
AXIS_VALUE_FOR_PER_TENSOR = 65535

def _scales_zp_are_constant(scales: torch.Tensor, zero_points: Optional[torch.Tensor], atol: float=1e-06) -> bool:
    s = scales.float().flatten()
    if s.numel() == 0:
        return True
    if (s.max() - s.min()).item() > atol:
        return False
    if zero_points is not None:
        z = zero_points.float().flatten()
        if z.numel() > 0 and (z.max() - z.min()).item() > atol:
            return False
    return True

def _layer_norm_ref(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps: float, axis: int=-1) -> torch.Tensor:
    if axis in (AXIS_VALUE_FOR_PER_TENSOR, AXIS_MUL_MODE):
        dim = -1
    else:
        dim = axis if axis >= 0 else x.dim() + axis
    mean = x.mean(dim=dim, keepdim=True)
    var = x.var(dim=dim, keepdim=True, unbiased=False) + eps
    rstd = torch.rsqrt(var)
    return (x - mean) * rstd * gamma + beta

def _quantize_per_tensor(x: torch.Tensor, scale: torch.Tensor, zero_point: Optional[torch.Tensor], div_mode: bool) -> torch.Tensor:
    """与 kernel 一致：div_mode 时 y=round(x/scale+zp)，mul 时 y=round(x*scale+zp)。"""
    if div_mode:
        q = x / scale
    else:
        q = x * scale
    if zero_point is not None:
        q = q + zero_point.float()
    return torch.round(q).clamp(-128, 127).to(torch.int8)

class Model(nn.Module):

    def __init__(self, gamma: torch.Tensor, beta: torch.Tensor, bias: torch.Tensor, scales: torch.Tensor, zero_points: Optional[torch.Tensor]=None, dtype_attr: int=0, axis: int=-1, epsilon: float=1e-05, additional_output: bool=False):
        super(Model, self).__init__()
        self.gamma = gamma
        self.beta = beta
        self.bias = bias
        self.scales = scales
        self.zero_points = zero_points
        self.dtype_attr = dtype_attr
        self.axis = axis
        self.epsilon = epsilon
        self.additional_output = additional_output

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> List[torch.Tensor]:
        dtype = x1.dtype
        x1_f = x1.float()
        x2_f = x2.float()
        x_sum = x1_f + x2_f + self.bias.float()
        x_norm = _layer_norm_ref(x_sum, self.gamma.float(), self.beta.float(), self.epsilon, self.axis)
        div_mode = self.axis != AXIS_MUL_MODE and self.axis != AXIS_VALUE_FOR_PER_TENSOR
        if self.axis == AXIS_VALUE_FOR_PER_TENSOR:
            use_per_tensor_scale_zp = True
        elif self.axis == AXIS_MUL_MODE:
            use_per_tensor_scale_zp = False
        else:
            assert self.axis == -1
            use_per_tensor_scale_zp = _scales_zp_are_constant(self.scales, self.zero_points)
        if use_per_tensor_scale_zp:
            scale_ref = self.scales.float().flatten()[0:1].expand_as(x_norm)
            zp_ref = self.zero_points.float().flatten()[0:1].expand_as(x_norm) if self.zero_points is not None else None
        else:
            scale_ref = self.scales.float()
            zp_ref = self.zero_points.float() if self.zero_points is not None else None
        y = _quantize_per_tensor(x_norm, scale_ref, zp_ref, div_mode)
        if self.axis == AXIS_VALUE_FOR_PER_TENSOR:
            x_out = x_sum.to(dtype)
        else:
            x_out = torch.round(x_sum).to(dtype)
        return [y, x_out]

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level2_134_QuantizeAddLayerNorm.json')
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
            x1 = torch.rand(x1_info["shape"], dtype=DTYPE_MAP[x1_info["dtype"]])
        if "data" in x2_info:
            x2 = torch.tensor(x2_info["data"], dtype=DTYPE_MAP[x2_info["dtype"]]).reshape(x2_info["shape"])
        else:
            x2 = torch.rand(x2_info["shape"], dtype=DTYPE_MAP[x2_info["dtype"]])

        input_groups.append([x1, x2])
    return input_groups


def get_init_inputs():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level2_134_QuantizeAddLayerNorm.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    init_groups = []
    for case in cases:
        entries = case.get("init_inputs", [])
        gamma_info = entries[0]
        beta_info = entries[1]
        bias_info = entries[2]
        scales_info = entries[3]
        zero_points_info = entries[4]
        dtype_attr_info = entries[5]
        axis_info = entries[6]
        epsilon_info = entries[7]
        additional_output_info = entries[8]
        if "data" in gamma_info:
            gamma = torch.tensor(gamma_info["data"], dtype=DTYPE_MAP[gamma_info["dtype"]]).reshape(gamma_info["shape"])
        else:
            gamma = torch.rand(gamma_info["shape"], dtype=DTYPE_MAP[gamma_info["dtype"]])
        if "data" in beta_info:
            beta = torch.tensor(beta_info["data"], dtype=DTYPE_MAP[beta_info["dtype"]]).reshape(beta_info["shape"])
        else:
            beta = torch.rand(beta_info["shape"], dtype=DTYPE_MAP[beta_info["dtype"]])
        if "data" in bias_info:
            bias = torch.tensor(bias_info["data"], dtype=DTYPE_MAP[bias_info["dtype"]]).reshape(bias_info["shape"])
        else:
            bias = torch.rand(bias_info["shape"], dtype=DTYPE_MAP[bias_info["dtype"]])
        if "data" in scales_info:
            scales = torch.tensor(scales_info["data"], dtype=DTYPE_MAP[scales_info["dtype"]]).reshape(scales_info["shape"])
        else:
            scales = torch.rand(scales_info["shape"], dtype=DTYPE_MAP[scales_info["dtype"]])
        if "data" in zero_points_info:
            zero_points = torch.tensor(zero_points_info["data"], dtype=DTYPE_MAP[zero_points_info["dtype"]]).reshape(zero_points_info["shape"])
        else:
            zero_points = torch.full(zero_points_info["shape"], zero_points_info["fill"], dtype=DTYPE_MAP[zero_points_info["dtype"]])
        dtype_attr = dtype_attr_info["value"]
        axis = axis_info["value"]
        epsilon = epsilon_info["value"]
        additional_output = additional_output_info["value"]
        init_groups.append([gamma, beta, bias, scales, zero_points, dtype_attr, axis, epsilon, additional_output])
    return init_groups
