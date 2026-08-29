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

class Model(torch.nn.Module):

    def __init__(self, unpermuted_tokens_grad, sorted_indices, permuted_probs_output_grad, topk_num, range_vals, padded_mode):
        super(Model, self).__init__()
        self.topk_num = topk_num
        self.range_vals = range_vals
        self.padded_mode = padded_mode

    def forward(self, unpermuted_tokens_grad, sorted_indices, permuted_probs_output_grad, topk_num, range_vals, padded_mode):
        hidden_size = unpermuted_tokens_grad.shape[1]
        num_tokens = sorted_indices.shape[0]
        token_grad_out = torch.zeros(num_tokens, hidden_size, dtype=unpermuted_tokens_grad.dtype, device=unpermuted_tokens_grad.device)
        sorted_indices_long = sorted_indices.to(torch.int64)
        if range_vals is not None and len(range_vals) == 2:
            start = range_vals[0]
            end = range_vals[1]
            mask = (sorted_indices_long >= start) & (sorted_indices_long < end)
            valid_indices = sorted_indices_long[mask] - start
            token_grad_out[mask] = unpermuted_tokens_grad[valid_indices]
        token_grad_out = token_grad_out.reshape(-1, topk_num, hidden_size)
        token_grad_out = token_grad_out.sum(dim=1)
        if permuted_probs_output_grad is not None:
            probs_grad_out = torch.zeros(num_tokens, topk_num, dtype=permuted_probs_output_grad.dtype, device=permuted_probs_output_grad.device)
            if range_vals is not None and len(range_vals) == 2:
                start = range_vals[0]
                end = range_vals[1]
                mask = (sorted_indices_long >= start) & (sorted_indices_long < end)
                valid_indices = sorted_indices_long[mask] - start
                probs_grad_out[mask] = permuted_probs_output_grad.view(-1, 1).expand(-1, topk_num)[valid_indices]
            if range_vals is not None and len(range_vals) == 2:
                probs_grad_out = probs_grad_out[range_vals[0]:range_vals[1]]
            probs_grad_out = probs_grad_out.reshape(-1, topk_num)
        else:
            probs_grad_out = None
        return [token_grad_out, probs_grad_out]

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level3_30_MoeTokenPermuteWithEpGrad.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        unpermuted_tokens_grad_info = inputs[0]
        sorted_indices_info = inputs[1]
        permuted_probs_output_grad_info = inputs[2]
        topk_num_info = inputs[3]
        range_vals_info = inputs[4]
        padded_mode_info = inputs[5]

        if "data" in unpermuted_tokens_grad_info:
            unpermuted_tokens_grad = torch.tensor(unpermuted_tokens_grad_info["data"], dtype=DTYPE_MAP[unpermuted_tokens_grad_info["dtype"]]).reshape(unpermuted_tokens_grad_info["shape"])
        else:
            unpermuted_tokens_grad = torch.rand(unpermuted_tokens_grad_info["shape"], dtype=DTYPE_MAP[unpermuted_tokens_grad_info["dtype"]]) * (unpermuted_tokens_grad_info["range"][1] - unpermuted_tokens_grad_info["range"][0]) + unpermuted_tokens_grad_info["range"][0]
        if "data" in sorted_indices_info:
            sorted_indices = torch.tensor(sorted_indices_info["data"], dtype=DTYPE_MAP[sorted_indices_info["dtype"]]).reshape(sorted_indices_info["shape"])
        else:
            sorted_indices = torch.randint(sorted_indices_info["range"][0], sorted_indices_info["range"][1] + 1, tuple(sorted_indices_info["shape"]), dtype=DTYPE_MAP[sorted_indices_info["dtype"]])
        if "data" in permuted_probs_output_grad_info:
            permuted_probs_output_grad = torch.tensor(permuted_probs_output_grad_info["data"], dtype=DTYPE_MAP[permuted_probs_output_grad_info["dtype"]]).reshape(permuted_probs_output_grad_info["shape"])
        else:
            permuted_probs_output_grad = torch.rand(permuted_probs_output_grad_info["shape"], dtype=DTYPE_MAP[permuted_probs_output_grad_info["dtype"]]) * (permuted_probs_output_grad_info["range"][1] - permuted_probs_output_grad_info["range"][0]) + permuted_probs_output_grad_info["range"][0]
        topk_num = topk_num_info["value"]
        range_vals = range_vals_info["value"]
        padded_mode = padded_mode_info["value"]

        input_groups.append([unpermuted_tokens_grad, sorted_indices, permuted_probs_output_grad, topk_num, range_vals, padded_mode])
    return input_groups


def get_init_inputs():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level3_30_MoeTokenPermuteWithEpGrad.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    init_groups = []
    for case in cases:
        entries = case.get("init_inputs", [])
        unpermuted_tokens_grad_info = entries[0]
        sorted_indices_info = entries[1]
        permuted_probs_output_grad_info = entries[2]
        topk_num_info = entries[3]
        range_vals_info = entries[4]
        padded_mode_info = entries[5]
        if "data" in unpermuted_tokens_grad_info:
            unpermuted_tokens_grad = torch.tensor(unpermuted_tokens_grad_info["data"], dtype=DTYPE_MAP[unpermuted_tokens_grad_info["dtype"]]).reshape(unpermuted_tokens_grad_info["shape"])
        else:
            unpermuted_tokens_grad = torch.rand(unpermuted_tokens_grad_info["shape"], dtype=DTYPE_MAP[unpermuted_tokens_grad_info["dtype"]]) * (unpermuted_tokens_grad_info["range"][1] - unpermuted_tokens_grad_info["range"][0]) + unpermuted_tokens_grad_info["range"][0]
        if "data" in sorted_indices_info:
            sorted_indices = torch.tensor(sorted_indices_info["data"], dtype=DTYPE_MAP[sorted_indices_info["dtype"]]).reshape(sorted_indices_info["shape"])
        else:
            sorted_indices = torch.randint(sorted_indices_info["range"][0], sorted_indices_info["range"][1] + 1, tuple(sorted_indices_info["shape"]), dtype=DTYPE_MAP[sorted_indices_info["dtype"]])
        if "data" in permuted_probs_output_grad_info:
            permuted_probs_output_grad = torch.tensor(permuted_probs_output_grad_info["data"], dtype=DTYPE_MAP[permuted_probs_output_grad_info["dtype"]]).reshape(permuted_probs_output_grad_info["shape"])
        else:
            permuted_probs_output_grad = torch.rand(permuted_probs_output_grad_info["shape"], dtype=DTYPE_MAP[permuted_probs_output_grad_info["dtype"]]) * (permuted_probs_output_grad_info["range"][1] - permuted_probs_output_grad_info["range"][0]) + permuted_probs_output_grad_info["range"][0]
        topk_num = topk_num_info["value"]
        range_vals = range_vals_info["value"]
        padded_mode = padded_mode_info["value"]
        init_groups.append([unpermuted_tokens_grad, sorted_indices, permuted_probs_output_grad, topk_num, range_vals, padded_mode])
    return init_groups
