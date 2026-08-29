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

def gelu_compute_erf(input_x: torch.Tensor) -> torch.Tensor:
    """
    Computes a GELU approximation using a polynomial approximation of the erf function.
    This implementation mirrors the provided numpy version for float32 precision.

    Args:
        input_x: A torch.Tensor representing the input.

    Returns:
        A torch.Tensor with the computed GELU approximation.
    """
    input_x = input_x.to(torch.float32)
    input_x_clamped_min = torch.max(input_x, torch.tensor(-13.25, dtype=torch.float32))
    x1 = torch.min(input_x_clamped_min, torch.tensor(5.75, dtype=torch.float32))
    x_pow = x1 * x1
    a1 = torch.tensor(-3.512339572e-09, dtype=torch.float32)
    a2 = torch.tensor(2.64526617e-07, dtype=torch.float32)
    a3 = torch.tensor(-7.929488134e-06, dtype=torch.float32)
    a4 = torch.tensor(0.000110612384, dtype=torch.float32)
    a5 = torch.tensor(6.518995814e-05, dtype=torch.float32)
    a6 = torch.tensor(-0.07266616915, dtype=torch.float32)
    a7 = torch.tensor(-1.595769883, dtype=torch.float32)
    y = x_pow * a1 + a2
    y = y * x_pow + a3
    y = y * x_pow + a4
    y = y * x_pow + a5
    y = y * x_pow + a6
    y = y * x_pow + a7
    y = y * x1
    y = torch.exp(y) + 1.0
    res = input_x / y
    return res

def tanh_parameter_compute(input_x: torch.Tensor) -> torch.Tensor:
    """
    Helper function to compute the x + 0.044715*x^3 term for the tanh GELU approximation.

    Args:
        input_x: A torch.Tensor representing the input.

    Returns:
        A torch.Tensor with the computed value.
    """
    input_x = input_x.to(torch.float32)
    y = input_x * input_x
    y = y * input_x
    y = y * torch.tensor(0.044715, dtype=torch.float32)
    result = input_x + y
    return result

def gelu_compute_tanh(input_x: torch.Tensor) -> torch.Tensor:
    """
    Computes a GELU approximation using the tanh formula:
    gelu(x) = x / (1 + exp(-sqrt(8/pi) * (x + 0.044715*x^3)))
    This implementation mirrors the provided numpy version for float32 precision.

    Args:
        input_x: A torch.Tensor representing the input.

    Returns:
        A torch.Tensor with the computed GELU approximation.
    """
    input_x = input_x.to(torch.float32)
    tanh_parameter = tanh_parameter_compute(input_x)
    mul_0 = tanh_parameter * torch.tensor(-1.5957691, dtype=torch.float32)
    temp = torch.exp(mul_0) + 1.0
    res = input_x / temp
    return res

class Model(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, scale: torch.Tensor, offset: torch.Tensor, approximate: str='tanh', quant_mode: str='static') -> List[torch.Tensor]:
        x_f = x.float()
        scale = scale.float()
        offset = offset.float()
        if approximate == 'none':
            gelu = gelu_compute_erf(x)
        else:
            gelu = gelu_compute_tanh(x)
        if scale.dim() == 1:
            scale = scale.view(*[1] * (x.dim() - 1), -1)
        if offset is not None and offset.dim() == 1:
            offset = offset.view(*[1] * (x.dim() - 1), -1)
        if quant_mode == 'static':
            quant = torch.round(gelu * scale + offset).clamp(-128, 127).to(torch.int8)
            return [quant]
        else:
            mul_res = gelu * scale
            max_abs = torch.amax(mul_res.abs(), dim=-1, keepdim=True)
            tmp_out_scale = 127.0 / (max_abs + 1e-06)
            out_scale = 1.0 / tmp_out_scale
            tmp_out_scale = tmp_out_scale.expand_as(mul_res)
            quant = torch.round(mul_res * tmp_out_scale).clamp(-128, 127).to(torch.int8)
            return [quant, out_scale]

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level2_105_GeluQuant.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        x_info = inputs[0]
        scale_info = inputs[1]
        offset_info = inputs[2]
        approximate_info = inputs[3]
        quant_mode_info = inputs[4]

        if "data" in x_info:
            x = torch.tensor(x_info["data"], dtype=DTYPE_MAP[x_info["dtype"]]).reshape(x_info["shape"])
        else:
            x = torch.rand(x_info["shape"], dtype=DTYPE_MAP[x_info["dtype"]]) * (x_info["range"][1] - x_info["range"][0]) + x_info["range"][0]
        if "data" in scale_info:
            scale = torch.tensor(scale_info["data"], dtype=DTYPE_MAP[scale_info["dtype"]]).reshape(scale_info["shape"])
        else:
            scale = torch.rand(scale_info["shape"], dtype=DTYPE_MAP[scale_info["dtype"]]) * (scale_info["range"][1] - scale_info["range"][0]) + scale_info["range"][0]
        if "data" in offset_info:
            offset = torch.tensor(offset_info["data"], dtype=DTYPE_MAP[offset_info["dtype"]]).reshape(offset_info["shape"])
        else:
            offset = torch.rand(offset_info["shape"], dtype=DTYPE_MAP[offset_info["dtype"]]) * (offset_info["range"][1] - offset_info["range"][0]) + offset_info["range"][0]
        approximate = approximate_info["value"]
        quant_mode = quant_mode_info["value"]

        input_groups.append([x, scale, offset, approximate, quant_mode])
    return input_groups


def get_init_inputs():
    return []
