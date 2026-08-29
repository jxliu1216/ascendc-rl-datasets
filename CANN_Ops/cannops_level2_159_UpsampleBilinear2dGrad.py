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
    """Golden model: use PyTorch autograd to compute the backward of bilinear 2d upsampling."""

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, grad_output, output_size, input_size, align_corners=False):
        dummy_input = torch.zeros(input_size, dtype=grad_output.dtype, device=grad_output.device, requires_grad=True)
        upsampled = F.interpolate(dummy_input, size=output_size[2:], mode='bilinear', align_corners=align_corners)
        upsampled.backward(grad_output)
        return dummy_input.grad

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level2_159_UpsampleBilinear2dGrad.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        grad_output_info = inputs[0]
        output_size_info = inputs[1]
        input_size_info = inputs[2]
        align_corners_info = inputs[3]

        if "data" in grad_output_info:
            grad_output = torch.tensor(grad_output_info["data"], dtype=DTYPE_MAP[grad_output_info["dtype"]]).reshape(grad_output_info["shape"])
        else:
            grad_output = torch.randn(grad_output_info["shape"], dtype=DTYPE_MAP[grad_output_info["dtype"]])
        output_size = output_size_info["value"]
        input_size = input_size_info["value"]
        align_corners = align_corners_info["value"]

        input_groups.append([grad_output, output_size, input_size, align_corners])
    return input_groups


def get_init_inputs():
    return []
