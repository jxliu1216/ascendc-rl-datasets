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

    def __init__(self, num_features: int, eps: float, momentum: float, affine: bool):
        super(Model, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine

    def forward(self, input_tensor: torch.Tensor, weight: Optional[torch.Tensor], bias: Optional[torch.Tensor], running_mean: torch.Tensor, running_var: torch.Tensor, training: bool) -> List[torch.Tensor]:
        reduction_dims = [0] + list(range(2, input_tensor.dim()))
        batch_mean = input_tensor.mean(dim=reduction_dims, keepdim=True)
        batch_variance = (input_tensor - batch_mean).pow(2).mean(dim=reduction_dims, keepdim=True)
        save_invstd = torch.rsqrt(batch_variance + self.eps)
        normalized_input = (input_tensor - batch_mean) * save_invstd
        output = normalized_input
        if weight is not None:
            output = output * weight.view(1, -1, *[1] * (input_tensor.dim() - 2))
        if bias is not None:
            output = output + bias.view(1, -1, *[1] * (input_tensor.dim() - 2))
        if training:
            with torch.no_grad():
                running_mean.copy_((1 - self.momentum) * running_mean + self.momentum * batch_mean.squeeze())
                running_var.copy_((1 - self.momentum) * running_var + self.momentum * batch_variance.squeeze())
        return [output, batch_mean.squeeze(), save_invstd.squeeze()]

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level2_19_BatchNormV3.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        input_tensor_info = inputs[0]
        weight_info = inputs[1]
        bias_info = inputs[2]
        running_mean_info = inputs[3]
        running_var_info = inputs[4]
        training_info = inputs[5]

        if "data" in input_tensor_info:
            input_tensor = torch.tensor(input_tensor_info["data"], dtype=DTYPE_MAP[input_tensor_info["dtype"]]).reshape(input_tensor_info["shape"])
        else:
            input_tensor = torch.rand(input_tensor_info["shape"], dtype=DTYPE_MAP[input_tensor_info["dtype"]])
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
        if "data" in running_mean_info:
            running_mean = torch.tensor(running_mean_info["data"], dtype=DTYPE_MAP[running_mean_info["dtype"]]).reshape(running_mean_info["shape"])
        else:
            running_mean = torch.rand(running_mean_info["shape"], dtype=DTYPE_MAP[running_mean_info["dtype"]])
        if "data" in running_var_info:
            running_var = torch.tensor(running_var_info["data"], dtype=DTYPE_MAP[running_var_info["dtype"]]).reshape(running_var_info["shape"])
        else:
            running_var = torch.rand(running_var_info["shape"], dtype=DTYPE_MAP[running_var_info["dtype"]]) * (running_var_info["range"][1] - running_var_info["range"][0]) + running_var_info["range"][0]
        training = training_info["value"]

        input_groups.append([input_tensor, weight, bias, running_mean, running_var, training])
    return input_groups


def get_init_inputs():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level2_19_BatchNormV3.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    init_groups = []
    for case in cases:
        entries = case.get("init_inputs", [])
        num_features_info = entries[0]
        eps_info = entries[1]
        momentum_info = entries[2]
        affine_info = entries[3]
        num_features = num_features_info["value"]
        eps = eps_info["value"]
        momentum = momentum_info["value"]
        affine = affine_info["value"]
        init_groups.append([num_features, eps, momentum, affine])
    return init_groups
