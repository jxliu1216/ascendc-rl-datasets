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

from typing import List, Tuple, Optional
import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self, num_features: int, eps: float):
        super(Model, self).__init__()
        self.num_features = num_features
        self.eps = eps

    def forward(self, x: torch.Tensor, mean: torch.Tensor, var: torch.Tensor, input_scale: torch.Tensor, input_zero_point: torch.Tensor, output_scale: torch.Tensor, output_zero_point: torch.Tensor, weight: Optional[torch.Tensor], bias: Optional[torch.Tensor]) -> torch.Tensor:
        x_fp32 = (x.float() - input_zero_point.item()) * input_scale.item()
        normalized = (x_fp32 - mean.unsqueeze(0).unsqueeze(2).unsqueeze(3)) / torch.sqrt(var.unsqueeze(0).unsqueeze(2).unsqueeze(3) + self.eps)
        if weight is not None:
            normalized = normalized * weight.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        if bias is not None:
            normalized = normalized + bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        y_fp32 = normalized / output_scale.item() + output_zero_point.item()
        y = y_fp32.round().clamp(-128, 127).to(x.dtype)
        return y

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level2_135_QuantizedBatchNorm.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        x_info = inputs[0]
        mean_info = inputs[1]
        var_info = inputs[2]
        input_scale_info = inputs[3]
        input_zero_point_info = inputs[4]
        output_scale_info = inputs[5]
        output_zero_point_info = inputs[6]
        weight_info = inputs[7]
        bias_info = inputs[8]

        if "data" in x_info:
            x = torch.tensor(x_info["data"], dtype=DTYPE_MAP[x_info["dtype"]]).reshape(x_info["shape"])
        else:
            x = torch.randint(x_info["range"][0], x_info["range"][1] + 1, tuple(x_info["shape"]), dtype=DTYPE_MAP[x_info["dtype"]])
        if "data" in mean_info:
            mean = torch.tensor(mean_info["data"], dtype=DTYPE_MAP[mean_info["dtype"]]).reshape(mean_info["shape"])
        else:
            mean = torch.rand(mean_info["shape"], dtype=DTYPE_MAP[mean_info["dtype"]])
        if "data" in var_info:
            var = torch.tensor(var_info["data"], dtype=DTYPE_MAP[var_info["dtype"]]).reshape(var_info["shape"])
        else:
            var = torch.rand(var_info["shape"], dtype=DTYPE_MAP[var_info["dtype"]])
        if "data" in input_scale_info:
            input_scale = torch.tensor(input_scale_info["data"], dtype=DTYPE_MAP[input_scale_info["dtype"]]).reshape(input_scale_info["shape"])
        else:
            input_scale = torch.full(input_scale_info["shape"], input_scale_info["fill"], dtype=DTYPE_MAP[input_scale_info["dtype"]])
        if "data" in input_zero_point_info:
            input_zero_point = torch.tensor(input_zero_point_info["data"], dtype=DTYPE_MAP[input_zero_point_info["dtype"]]).reshape(input_zero_point_info["shape"])
        else:
            input_zero_point = torch.full(input_zero_point_info["shape"], input_zero_point_info["fill"], dtype=DTYPE_MAP[input_zero_point_info["dtype"]])
        if "data" in output_scale_info:
            output_scale = torch.tensor(output_scale_info["data"], dtype=DTYPE_MAP[output_scale_info["dtype"]]).reshape(output_scale_info["shape"])
        else:
            output_scale = torch.full(output_scale_info["shape"], output_scale_info["fill"], dtype=DTYPE_MAP[output_scale_info["dtype"]])
        if "data" in output_zero_point_info:
            output_zero_point = torch.tensor(output_zero_point_info["data"], dtype=DTYPE_MAP[output_zero_point_info["dtype"]]).reshape(output_zero_point_info["shape"])
        else:
            output_zero_point = torch.full(output_zero_point_info["shape"], output_zero_point_info["fill"], dtype=DTYPE_MAP[output_zero_point_info["dtype"]])
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
        if bias_info["type"] == "attr":
            if bias_info.get("dtype") == "none":
                bias = None
            else:
                bias = bias_info["value"]
        else:
            if "data" in bias_info:
                bias = torch.tensor(bias_info["data"], dtype=DTYPE_MAP[bias_info["dtype"]]).reshape(bias_info["shape"])
            else:
                bias = torch.rand(bias_info["shape"], dtype=DTYPE_MAP[bias_info["dtype"]])

        input_groups.append([x, mean, var, input_scale, input_zero_point, output_scale, output_zero_point, weight, bias])
    return input_groups


def get_init_inputs():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level2_135_QuantizedBatchNorm.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    init_groups = []
    for case in cases:
        entries = case.get("init_inputs", [])
        num_features_info = entries[0]
        eps_info = entries[1]
        num_features = num_features_info["value"]
        eps = eps_info["value"]
        init_groups.append([num_features, eps])
    return init_groups
