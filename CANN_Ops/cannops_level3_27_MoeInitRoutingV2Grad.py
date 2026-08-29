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
import numpy as np

def moe_init_routing_v2_grad(grad_expanded_x, expanded_row_idx, top_k, drop_pad_mode, active_num):
    num_rows = grad_expanded_x.shape[0] // top_k
    hidden_size = grad_expanded_x.shape[-1]
    grad_x = torch.zeros((num_rows, hidden_size), dtype=torch.float32)
    for i in range(num_rows):
        for j in range(i * top_k, i * top_k + top_k, 1):
            expanded_x_idx = expanded_row_idx[j]
            if drop_pad_mode == 1:
                if expended_x_idx == -1:
                    continue
            elif active_num > 0:
                if expanded_x_idx >= active_num:
                    continue
            grad_x[i] = torch.add(grad_x[i], grad_expanded_x[expanded_x_idx].to(torch.float32))
    return grad_x

class Model(torch.nn.Module):

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, grad_expanded_x, expanded_row_idx, top_k, drop_pad_mode, active_num):
        if grad_expanded_x.dtype == torch.bfloat16:
            grad_expanded_x = grad_expanded_x.float()
        expanded_row_idx = expanded_row_idx
        output = moe_init_routing_v2_grad(grad_expanded_x, expanded_row_idx, top_k, drop_pad_mode, active_num)
        return output.to(grad_expanded_x.dtype)

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level3_27_MoeInitRoutingV2Grad.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        grad_expanded_x_info = inputs[0]
        expanded_row_idx_info = inputs[1]
        top_k_info = inputs[2]
        drop_pad_mode_info = inputs[3]
        active_num_info = inputs[4]

        if "data" in grad_expanded_x_info:
            grad_expanded_x = torch.tensor(grad_expanded_x_info["data"], dtype=DTYPE_MAP[grad_expanded_x_info["dtype"]]).reshape(grad_expanded_x_info["shape"])
        else:
            grad_expanded_x = torch.randn(grad_expanded_x_info["shape"], dtype=DTYPE_MAP[grad_expanded_x_info["dtype"]])
        if "data" in expanded_row_idx_info:
            expanded_row_idx = torch.tensor(expanded_row_idx_info["data"], dtype=DTYPE_MAP[expanded_row_idx_info["dtype"]]).reshape(expanded_row_idx_info["shape"])
        else:
            expanded_row_idx = torch.randint(expanded_row_idx_info["range"][0], expanded_row_idx_info["range"][1] + 1, tuple(expanded_row_idx_info["shape"]), dtype=DTYPE_MAP[expanded_row_idx_info["dtype"]])
        top_k = top_k_info["value"]
        drop_pad_mode = drop_pad_mode_info["value"]
        active_num = active_num_info["value"]

        input_groups.append([grad_expanded_x, expanded_row_idx, top_k, drop_pad_mode, active_num])
    return input_groups


def get_init_inputs():
    return []
