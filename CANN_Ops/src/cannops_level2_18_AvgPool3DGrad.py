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
import math

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, grad_output: torch.Tensor, x: torch.Tensor, ksize: List[int], strides: List[int], pads: List[int], ceil_mode: bool, count_include_pad: bool, divisor_override: int, data_format: str) -> torch.Tensor:
        res = torch.ops.aten.avg_pool3d_backward(grad_output.to(torch.float32), x.to(torch.float32), ksize, strides, pads, ceil_mode, count_include_pad, divisor_override)
        return res.to(x.dtype)

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level2_18_AvgPool3DGrad.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        grad_output_info = inputs[0]
        x_info = inputs[1]
        ksize_info = inputs[2]
        strides_info = inputs[3]
        pads_info = inputs[4]
        ceil_mode_info = inputs[5]
        count_include_pad_info = inputs[6]
        divisor_override_info = inputs[7]
        data_format_info = inputs[8]

        if "data" in grad_output_info:
            grad_output = torch.tensor(grad_output_info["data"], dtype=DTYPE_MAP[grad_output_info["dtype"]]).reshape(grad_output_info["shape"])
        else:
            grad_output = torch.randn(grad_output_info["shape"], dtype=DTYPE_MAP[grad_output_info["dtype"]])
        if "data" in x_info:
            x = torch.tensor(x_info["data"], dtype=DTYPE_MAP[x_info["dtype"]]).reshape(x_info["shape"])
        else:
            x = torch.randn(x_info["shape"], dtype=DTYPE_MAP[x_info["dtype"]])
        ksize = ksize_info["value"]
        strides = strides_info["value"]
        pads = pads_info["value"]
        ceil_mode = ceil_mode_info["value"]
        count_include_pad = count_include_pad_info["value"]
        divisor_override = divisor_override_info["value"]
        data_format = data_format_info["value"]

        input_groups.append([grad_output, x, ksize, strides, pads, ceil_mode, count_include_pad, divisor_override, data_format])
    return input_groups


def get_init_inputs():
    return []
