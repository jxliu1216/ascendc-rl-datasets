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

    def forward(self, x: torch.Tensor, ksize: List[int], strides: List[int], pads: List[int], dilation: List[int], ceil_mode: bool, data_format: str) -> torch.Tensor:
        m = torch.nn.MaxPool3d(kernel_size=ksize, stride=strides, padding=pads, dilation=dilation, return_indices=True, ceil_mode=ceil_mode)
        ori_dtype = x.dtype
        if x.dtype != torch.float:
            x_float = x.to(torch.float)
        else:
            x_float = x
        output, indices = m(x_float)
        if ori_dtype != torch.float:
            output = output.to(ori_dtype)
        indices = indices.to(torch.int32)
        return [output.cpu(), indices.cpu()]

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level2_127_MaxPool3DWithArgmaxV2.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        x_info = inputs[0]
        ksize_info = inputs[1]
        strides_info = inputs[2]
        pads_info = inputs[3]
        dilation_info = inputs[4]
        ceil_mode_info = inputs[5]
        data_format_info = inputs[6]

        if "data" in x_info:
            x = torch.tensor(x_info["data"], dtype=DTYPE_MAP[x_info["dtype"]]).reshape(x_info["shape"])
        else:
            x = torch.randn(x_info["shape"], dtype=DTYPE_MAP[x_info["dtype"]])
        ksize = ksize_info["value"]
        strides = strides_info["value"]
        pads = pads_info["value"]
        dilation = dilation_info["value"]
        ceil_mode = ceil_mode_info["value"]
        data_format = data_format_info["value"]

        input_groups.append([x, ksize, strides, pads, dilation, ceil_mode, data_format])
    return input_groups


def get_init_inputs():
    return []
