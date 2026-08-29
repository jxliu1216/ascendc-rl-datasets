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

def _pytorch_pad_tuple_from_ge_eight(ge_pad):
    """ge_pad: [0,0,0,0, top, bottom, left, right] -> F.pad 用 (left, right, top, bottom)。"""
    pt, pb, pl, pr = (int(ge_pad[4]), int(ge_pad[5]), int(ge_pad[6]), int(ge_pad[7]))
    return (pl, pr, pt, pb)

class Model(nn.Module):
    """CPU 参考：replication pad 反向；半精度在 CPU 上无 replicate 实现，先 float32 反传再 cast 回原 dtype。"""

    def __init__(self):
        super().__init__()

    def forward(self, grad_output: torch.Tensor, ge_padding):
        pad = _pytorch_pad_tuple_from_ge_eight(ge_padding)
        nd = grad_output.dim()
        if nd != 4:
            raise ValueError('PadV3GradReplicate validation 仅支持 4D NCHW')
        h = grad_output.shape[2] - pad[2] - pad[3]
        w = grad_output.shape[3] - pad[0] - pad[1]
        self_shape = (grad_output.shape[0], grad_output.shape[1], h, w)
        orig_dtype = grad_output.dtype
        g = grad_output.to(dtype=torch.float32)
        grad_input = torch.zeros(self_shape, dtype=torch.float32, device=g.device, requires_grad=True)
        out = F.pad(grad_input, pad, mode='replicate')
        out.backward(g)
        return grad_input.grad.to(dtype=orig_dtype)

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level1_49_PadV3GradReplicate.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        grad_output_info = inputs[0]
        ge_padding_info = inputs[1]

        if "data" in grad_output_info:
            grad_output = torch.tensor(grad_output_info["data"], dtype=DTYPE_MAP[grad_output_info["dtype"]]).reshape(grad_output_info["shape"])
        else:
            grad_output = torch.randn(grad_output_info["shape"], dtype=DTYPE_MAP[grad_output_info["dtype"]])
        ge_padding = ge_padding_info["value"]

        input_groups.append([grad_output, ge_padding])
    return input_groups


def get_init_inputs():
    return []
