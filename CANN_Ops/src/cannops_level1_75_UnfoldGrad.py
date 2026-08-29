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

class Model(nn.Module):

    def __init__(self, input_shape: List[int], dim: int, size: int, step: int):
        super().__init__()
        self.input_shape = list(input_shape)
        self.dim = int(dim)
        self.size = int(size)
        self.step = int(step)

    def forward(self, grad_out: torch.Tensor) -> torch.Tensor:
        x = torch.zeros(self.input_shape, dtype=grad_out.dtype, device=grad_out.device, requires_grad=True)
        u = x.unfold(self.dim, self.size, self.step)
        u.backward(grad_out)
        return x.grad

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level1_75_UnfoldGrad.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        grad_out_info = inputs[0]

        if "data" in grad_out_info:
            grad_out = torch.tensor(grad_out_info["data"], dtype=DTYPE_MAP[grad_out_info["dtype"]]).reshape(grad_out_info["shape"])
        else:
            grad_out = torch.randn(grad_out_info["shape"], dtype=DTYPE_MAP[grad_out_info["dtype"]])

        input_groups.append([grad_out])
    return input_groups


def get_init_inputs():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level1_75_UnfoldGrad.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    init_groups = []
    for case in cases:
        entries = case.get("init_inputs", [])
        input_shape_info = entries[0]
        dim_info = entries[1]
        size_info = entries[2]
        step_info = entries[3]
        input_shape = input_shape_info["value"]
        dim = dim_info["value"]
        size = size_info["value"]
        step = step_info["value"]
        init_groups.append([input_shape, dim, size, step])
    return init_groups
