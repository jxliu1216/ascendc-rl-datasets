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
import torch.nn.functional as F

class Model(nn.Module):
    """CPU 参考：与 ops ST executor 一致，对 circular pad 做反向（grad 对 unpadded 输入）。"""

    def __init__(self):
        super().__init__()

    def forward(self, grad_output: torch.Tensor, padding):
        pad = tuple((int(p) for p in padding))
        nd = grad_output.dim()
        if len(pad) == 4:
            h = grad_output.shape[nd - 2] - pad[2] - pad[3]
            w = grad_output.shape[nd - 1] - pad[0] - pad[1]
            self_shape = tuple(grad_output.shape[:-2]) + (h, w)
        else:
            d = grad_output.shape[nd - 3] - pad[4] - pad[5]
            h = grad_output.shape[nd - 2] - pad[2] - pad[3]
            w = grad_output.shape[nd - 1] - pad[0] - pad[1]
            self_shape = tuple(grad_output.shape[:-3]) + (d, h, w)
        grad_input = torch.zeros(self_shape, dtype=grad_output.dtype, device=grad_output.device, requires_grad=True)
        out = F.pad(grad_input, pad, mode='circular')
        out.backward(grad_output)
        return grad_input.grad

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level1_11_CircularPadGrad.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        grad_output_info = inputs[0]
        padding_info = inputs[1]

        if "data" in grad_output_info:
            grad_output = torch.tensor(grad_output_info["data"], dtype=DTYPE_MAP[grad_output_info["dtype"]]).reshape(grad_output_info["shape"])
        else:
            grad_output = torch.randn(grad_output_info["shape"], dtype=DTYPE_MAP[grad_output_info["dtype"]])
        padding = padding_info["value"]

        input_groups.append([grad_output, padding])
    return input_groups


def get_init_inputs():
    return []
