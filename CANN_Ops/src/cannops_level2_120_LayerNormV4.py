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

class Model(nn.Module):

    def __init__(self, normalized_shape: List[int], weight: Optional[torch.Tensor], bias: Optional[torch.Tensor], eps: float):
        super(Model, self).__init__()
        self.normalized_shape = normalized_shape
        self.weight = weight
        self.bias = bias
        self.eps = eps

    def forward(self, input_tensor: torch.Tensor) -> List[torch.Tensor]:
        input_fp32 = input_tensor.to(torch.float32)
        weight_fp32 = self.weight.to(torch.float32) if self.weight is not None else None
        bias_fp32 = self.bias.to(torch.float32) if self.bias is not None else None
        input_dim = input_fp32.dim()
        normalized_dim = len(self.normalized_shape)
        reduction_dims = tuple(range(input_dim - normalized_dim, input_dim))
        mean = input_fp32.mean(dim=reduction_dims, keepdim=True)
        variance = (input_fp32 - mean).pow(2).mean(dim=reduction_dims, keepdim=True)
        rstd = torch.rsqrt(variance + self.eps)
        out = (input_fp32 - mean) * rstd
        if weight_fp32 is not None:
            out = out * weight_fp32
        if bias_fp32 is not None:
            out = out + bias_fp32
        out = out.to(input_tensor.dtype)
        mean = mean.to(input_tensor.dtype)
        rstd = rstd.to(input_tensor.dtype)
        return [out, mean, rstd]

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level2_120_LayerNormV4.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        input_tensor_info = inputs[0]

        if "data" in input_tensor_info:
            input_tensor = torch.tensor(input_tensor_info["data"], dtype=DTYPE_MAP[input_tensor_info["dtype"]]).reshape(input_tensor_info["shape"])
        else:
            input_tensor = torch.rand(input_tensor_info["shape"], dtype=DTYPE_MAP[input_tensor_info["dtype"]])

        input_groups.append([input_tensor])
    return input_groups


def get_init_inputs():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level2_120_LayerNormV4.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    init_groups = []
    for case in cases:
        entries = case.get("init_inputs", [])
        normalized_shape_info = entries[0]
        weight_info = entries[1]
        bias_info = entries[2]
        eps_info = entries[3]
        normalized_shape = normalized_shape_info["value"]
        if weight_info["type"] == "attr":
            if weight_info.get("dtype") == "none":
                weight = None
            else:
                weight = weight_info["value"]
        else:
            if "data" in weight_info:
                weight = torch.tensor(weight_info["data"], dtype=DTYPE_MAP[weight_info["dtype"]]).reshape(weight_info["shape"])
            else:
                weight = torch.rand(weight_info["shape"], dtype=DTYPE_MAP[weight_info["dtype"]])
        if "data" in bias_info:
            bias = torch.tensor(bias_info["data"], dtype=DTYPE_MAP[bias_info["dtype"]]).reshape(bias_info["shape"])
        else:
            bias = torch.rand(bias_info["shape"], dtype=DTYPE_MAP[bias_info["dtype"]])
        eps = eps_info["value"]
        init_groups.append([normalized_shape, weight, bias, eps])
    return init_groups
