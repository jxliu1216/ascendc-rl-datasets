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

class Model(torch.nn.Module):

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, indices: torch.Tensor, probs: torch.Tensor, range_vals: list, num_token_out: int, pad_mode: bool):
        if indices.dim() == 1:
            topk = 1
        else:
            topk = indices.size(1)
        flatten_indices = indices.view(-1)
        sorted_indices = torch.argsort(flatten_indices.float(), stable=True)
        sorted_indices1 = torch.argsort(sorted_indices.float(), stable=True)
        sorted_indices1 = sorted_indices1.to(torch.int32)
        if range_vals is not None:
            start = range_vals[0]
            end = range_vals[1]
            sorted_indices_sliced = sorted_indices[start:end]
        else:
            sorted_indices_sliced = sorted_indices
        permuted_tokens = x.index_select(0, sorted_indices_sliced // topk)
        if probs is not None:
            flatten_probs = probs.view(-1)
            permuted_probs = flatten_probs.index_select(0, sorted_indices_sliced)
        else:
            permuted_probs = torch.empty(0, device=x.device, dtype=probs.dtype) if probs is not None else None
        return [permuted_tokens, sorted_indices1, permuted_probs]

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level3_29_MoeTokenPermuteWithEp.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        x_info = inputs[0]
        indices_info = inputs[1]
        probs_info = inputs[2]
        range_vals_info = inputs[3]
        num_token_out_info = inputs[4]
        pad_mode_info = inputs[5]

        if "data" in x_info:
            x = torch.tensor(x_info["data"], dtype=DTYPE_MAP[x_info["dtype"]]).reshape(x_info["shape"])
        else:
            x = torch.randn(x_info["shape"], dtype=DTYPE_MAP[x_info["dtype"]])
        if "data" in indices_info:
            indices = torch.tensor(indices_info["data"], dtype=DTYPE_MAP[indices_info["dtype"]]).reshape(indices_info["shape"])
        else:
            indices = torch.randint(indices_info["range"][0], indices_info["range"][1] + 1, tuple(indices_info["shape"]), dtype=DTYPE_MAP[indices_info["dtype"]])
        if "data" in probs_info:
            probs = torch.tensor(probs_info["data"], dtype=DTYPE_MAP[probs_info["dtype"]]).reshape(probs_info["shape"])
        else:
            probs = torch.rand(probs_info["shape"], dtype=DTYPE_MAP[probs_info["dtype"]])
        range_vals = range_vals_info["value"]
        num_token_out = num_token_out_info["value"]
        pad_mode = pad_mode_info["value"]

        input_groups.append([x, indices, probs, range_vals, num_token_out, pad_mode])
    return input_groups


def get_init_inputs():
    return []
