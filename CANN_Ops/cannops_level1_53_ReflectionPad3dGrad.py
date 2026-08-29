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

import torch
import torch.nn as nn

class Model(nn.Module):
    """CPU 金标准：ReflectionPad3d 前向 + backward，得到对 self 的梯度。"""

    def __init__(self, padding):
        super().__init__()
        self.pad = nn.ReflectionPad3d(padding)

    def forward(self, grad_output: torch.Tensor, self_input: torch.Tensor) -> torch.Tensor:
        x = self_input.detach().clone().requires_grad_(True)
        y = self.pad(x)
        y.backward(grad_output)
        return x.grad

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level1_53_ReflectionPad3dGrad.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        grad_output_info = inputs[0]
        self_input_info = inputs[1]

        if "data" in grad_output_info:
            grad_output = torch.tensor(grad_output_info["data"], dtype=DTYPE_MAP[grad_output_info["dtype"]]).reshape(grad_output_info["shape"])
        else:
            grad_output = torch.randn(grad_output_info["shape"], dtype=DTYPE_MAP[grad_output_info["dtype"]])
        if "data" in self_input_info:
            self_input = torch.tensor(self_input_info["data"], dtype=DTYPE_MAP[self_input_info["dtype"]]).reshape(self_input_info["shape"])
        else:
            self_input = torch.randn(self_input_info["shape"], dtype=DTYPE_MAP[self_input_info["dtype"]])

        input_groups.append([grad_output, self_input])
    return input_groups


def get_init_inputs():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level1_53_ReflectionPad3dGrad.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    init_groups = []
    for case in cases:
        entries = case.get("init_inputs", [])
        padding_info = entries[0]
        padding = padding_info["value"]
        init_groups.append([padding])
    return init_groups
