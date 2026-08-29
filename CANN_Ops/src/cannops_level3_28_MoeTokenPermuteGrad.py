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

def permute(tokens, indices, num_out_tokens=None, padded_mode=False):
    if padded_mode:
        return (tokens.index_select(dim=0, index=indices.view(-1)), indices)
    if indices.dim() == 1:
        topk = 1
    else:
        topk = indices.size(1)
    flatten_indices = indices.view(-1)
    sorted_indices = torch.argsort(flatten_indices, stable=True)
    sorted_indices1 = torch.argsort(sorted_indices, stable=True)
    if num_out_tokens is not None and num_out_tokens != 0:
        sorted_indices = sorted_indices[:num_out_tokens]
    s_k = sorted_indices // topk
    permuted_tokens = tokens.index_select(0, s_k)
    return (permuted_tokens, sorted_indices1)

class Model(torch.nn.Module):

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, tokens, permuted_output_grad, indices, num_topk, padded_mode=False):
        tokens.requires_grad_(True)
        permuted_tokens, sorted_indices = permute(tokens, indices, num_topk, padded_mode)
        permuted_tokens.backward(permuted_output_grad)
        return tokens.grad

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level3_28_MoeTokenPermuteGrad.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        tokens_info = inputs[0]
        permuted_output_grad_info = inputs[1]
        indices_info = inputs[2]
        num_topk_info = inputs[3]
        padded_mode_info = inputs[4]

        if "data" in tokens_info:
            tokens = torch.tensor(tokens_info["data"], dtype=DTYPE_MAP[tokens_info["dtype"]]).reshape(tokens_info["shape"])
        else:
            tokens = torch.rand(tokens_info["shape"], dtype=DTYPE_MAP[tokens_info["dtype"]])
        if "data" in permuted_output_grad_info:
            permuted_output_grad = torch.tensor(permuted_output_grad_info["data"], dtype=DTYPE_MAP[permuted_output_grad_info["dtype"]]).reshape(permuted_output_grad_info["shape"])
        else:
            permuted_output_grad = torch.rand(permuted_output_grad_info["shape"], dtype=DTYPE_MAP[permuted_output_grad_info["dtype"]])
        if "data" in indices_info:
            indices = torch.tensor(indices_info["data"], dtype=DTYPE_MAP[indices_info["dtype"]]).reshape(indices_info["shape"])
        else:
            indices = torch.randint(indices_info["range"][0], indices_info["range"][1] + 1, tuple(indices_info["shape"]), dtype=DTYPE_MAP[indices_info["dtype"]])
        num_topk = num_topk_info["value"]
        padded_mode = padded_mode_info["value"]

        input_groups.append([tokens, permuted_output_grad, indices, num_topk, padded_mode])
    return input_groups


def get_init_inputs():
    return []
