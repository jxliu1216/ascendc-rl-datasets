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

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, softmax_output: torch.Tensor, grad_output: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        grad_softmax = torch.matmul(grad_output, torch.transpose(values, -2, -1))
        output = (grad_softmax - (softmax_output * grad_softmax).sum(-1, keepdim=True)) * softmax_output
        return output

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level2_118_InplaceFusedMatmulSoftmaxGrad.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        softmax_output_info = inputs[0]
        grad_output_info = inputs[1]
        values_info = inputs[2]

        if "data" in softmax_output_info:
            softmax_output = torch.tensor(softmax_output_info["data"], dtype=DTYPE_MAP[softmax_output_info["dtype"]]).reshape(softmax_output_info["shape"])
        else:
            softmax_output = torch.rand(softmax_output_info["shape"], dtype=DTYPE_MAP[softmax_output_info["dtype"]]) * (softmax_output_info["range"][1] - softmax_output_info["range"][0]) + softmax_output_info["range"][0]
        if "data" in grad_output_info:
            grad_output = torch.tensor(grad_output_info["data"], dtype=DTYPE_MAP[grad_output_info["dtype"]]).reshape(grad_output_info["shape"])
        else:
            grad_output = torch.rand(grad_output_info["shape"], dtype=DTYPE_MAP[grad_output_info["dtype"]]) * (grad_output_info["range"][1] - grad_output_info["range"][0]) + grad_output_info["range"][0]
        if "data" in values_info:
            values = torch.tensor(values_info["data"], dtype=DTYPE_MAP[values_info["dtype"]]).reshape(values_info["shape"])
        else:
            values = torch.rand(values_info["shape"], dtype=DTYPE_MAP[values_info["dtype"]]) * (values_info["range"][1] - values_info["range"][0]) + values_info["range"][0]

        input_groups.append([softmax_output, grad_output, values])
    return input_groups


def get_init_inputs():
    return []
