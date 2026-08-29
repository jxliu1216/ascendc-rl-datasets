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
    """PyTorch native reference implementation (golden model)."""

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, permuted_tokens, sorted_indices, probs, num_topk, range_start, range_end):
        """
        Args:
            permuted_tokens: [num_local_tokens, hidden_size] - tokens in the local EP partition
            sorted_indices: [num_tokens * top_k] - indices into global token positions
            probs: [num_tokens, top_k] or None - expert probabilities
            num_topk: number of top-k experts
            range_start: start of EP range
            range_end: end of EP range
        Returns:
            unpermuted_tokens: [num_tokens, hidden_size]
        """
        orig_dtype = permuted_tokens.dtype
        num_total = sorted_indices.shape[0]
        num_tokens = num_total // num_topk
        hidden_size = permuted_tokens.shape[1]
        tokens_f = permuted_tokens.float()
        output = torch.zeros(num_tokens, hidden_size, dtype=torch.float32, device=permuted_tokens.device)
        indices_cpu = sorted_indices.cpu()
        tokens_cpu = tokens_f.cpu()
        if probs is not None:
            probs_cpu = probs.float().cpu()
        for i in range(num_tokens):
            for j in range(num_topk):
                idx = indices_cpu[i * num_topk + j].item()
                if range_start <= idx < range_end:
                    token = tokens_cpu[idx - range_start]
                    if probs is not None:
                        prob_val = probs_cpu[i, j].item()
                        token = token * prob_val
                    output[i] += token
        return output.to(orig_dtype)

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level3_33_MoeTokenUnpermuteWithEp.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        permuted_tokens_info = inputs[0]
        sorted_indices_info = inputs[1]
        probs_info = inputs[2]
        num_topk_info = inputs[3]
        range_start_info = inputs[4]
        range_end_info = inputs[5]

        if "data" in permuted_tokens_info:
            permuted_tokens = torch.tensor(permuted_tokens_info["data"], dtype=DTYPE_MAP[permuted_tokens_info["dtype"]]).reshape(permuted_tokens_info["shape"])
        else:
            permuted_tokens = torch.randn(permuted_tokens_info["shape"], dtype=DTYPE_MAP[permuted_tokens_info["dtype"]]) * permuted_tokens_info["std"] + permuted_tokens_info["mean"]
        if "data" in sorted_indices_info:
            sorted_indices = torch.tensor(sorted_indices_info["data"], dtype=DTYPE_MAP[sorted_indices_info["dtype"]]).reshape(sorted_indices_info["shape"])
        else:
            sorted_indices = torch.randint(sorted_indices_info["range"][0], sorted_indices_info["range"][1] + 1, tuple(sorted_indices_info["shape"]), dtype=DTYPE_MAP[sorted_indices_info["dtype"]])
        if probs_info["type"] == "attr":
            if probs_info.get("dtype") == "none":
                probs = None
            else:
                probs = probs_info["value"]
        else:
            if "data" in probs_info:
                probs = torch.tensor(probs_info["data"], dtype=DTYPE_MAP[probs_info["dtype"]]).reshape(probs_info["shape"])
            else:
                probs = torch.rand(probs_info["shape"], dtype=DTYPE_MAP[probs_info["dtype"]])
        num_topk = num_topk_info["value"]
        range_start = range_start_info["value"]
        range_end = range_end_info["value"]

        input_groups.append([permuted_tokens, sorted_indices, probs, num_topk, range_start, range_end])
    return input_groups


def get_init_inputs():
    return []
