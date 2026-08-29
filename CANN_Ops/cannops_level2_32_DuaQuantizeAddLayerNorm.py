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

def _layer_norm_ref(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps: float, axis: int=-1) -> torch.Tensor:
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

    def __init__(self, gamma: torch.Tensor, beta: torch.Tensor, bias: torch.Tensor, scales1: torch.Tensor, scales2: torch.Tensor, zero_points1: Optional[torch.Tensor]=None, zero_points2: Optional[torch.Tensor]=None, dtype_attr: int=0, axis: int=-1, epsilon: float=1e-05, additional_output: bool=False):
        super(Model, self).__init__()
        self.gamma = gamma
        self.beta = beta
        self.bias = bias
        self.scales1 = scales1
        self.scales2 = scales2
        self.zero_points1 = zero_points1
        self.zero_points2 = zero_points2
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
        div_mode = self.axis != AXIS_MUL_MODE
        zp1 = self.zero_points1.float() if self.zero_points1 is not None else None
        zp2 = self.zero_points2.float() if self.zero_points2 is not None else None
        y1 = _quantize_per_tensor(x_norm, self.scales1.float(), zp1, div_mode)
        y2 = _quantize_per_tensor(x_norm, self.scales2.float(), zp2, div_mode)
        x_out = torch.round(x_sum).to(dtype)
        return [y1, y2, x_out]

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level2_32_DuaQuantizeAddLayerNorm.json')
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
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level2_32_DuaQuantizeAddLayerNorm.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    init_groups = []
    for case in cases:
        entries = case.get("init_inputs", [])
        gamma_info = entries[0]
        beta_info = entries[1]
        bias_info = entries[2]
        scales1_info = entries[3]
        scales2_info = entries[4]
        zero_points1_info = entries[5]
        zero_points2_info = entries[6]
        dtype_attr_info = entries[7]
        axis_info = entries[8]
        epsilon_info = entries[9]
        additional_output_info = entries[10]
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
        if "data" in scales1_info:
            scales1 = torch.tensor(scales1_info["data"], dtype=DTYPE_MAP[scales1_info["dtype"]]).reshape(scales1_info["shape"])
        else:
            scales1 = torch.rand(scales1_info["shape"], dtype=DTYPE_MAP[scales1_info["dtype"]])
        if "data" in scales2_info:
            scales2 = torch.tensor(scales2_info["data"], dtype=DTYPE_MAP[scales2_info["dtype"]]).reshape(scales2_info["shape"])
        else:
            scales2 = torch.rand(scales2_info["shape"], dtype=DTYPE_MAP[scales2_info["dtype"]])
        if "data" in zero_points1_info:
            zero_points1 = torch.tensor(zero_points1_info["data"], dtype=DTYPE_MAP[zero_points1_info["dtype"]]).reshape(zero_points1_info["shape"])
        else:
            zero_points1 = torch.full(zero_points1_info["shape"], zero_points1_info["fill"], dtype=DTYPE_MAP[zero_points1_info["dtype"]])
        if "data" in zero_points2_info:
            zero_points2 = torch.tensor(zero_points2_info["data"], dtype=DTYPE_MAP[zero_points2_info["dtype"]]).reshape(zero_points2_info["shape"])
        else:
            zero_points2 = torch.full(zero_points2_info["shape"], zero_points2_info["fill"], dtype=DTYPE_MAP[zero_points2_info["dtype"]])
        dtype_attr = dtype_attr_info["value"]
        axis = axis_info["value"]
        epsilon = epsilon_info["value"]
        additional_output = additional_output_info["value"]
        init_groups.append([gamma, beta, bias, scales1, scales2, zero_points1, zero_points2, dtype_attr, axis, epsilon, additional_output])
    return init_groups
