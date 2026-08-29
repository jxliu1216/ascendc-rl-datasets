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
    """PyTorch 原生算子参考实现（golden model）。"""

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, unpermuted_tokens_grad: torch.Tensor, sorted_indices: torch.Tensor, permuted_tokens: Optional[torch.Tensor], probs: Optional[torch.Tensor], padded_mode: bool, restore_shape: List[int], range_vals: List[int], topk_num: int) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens_num = unpermuted_tokens_grad.shape[0]
        hidden_size = unpermuted_tokens_grad.shape[1]
        total_indices = sorted_indices.shape[0]
        start = range_vals[0] if range_vals[0] >= 0 else 0
        end = range_vals[1] if range_vals[1] >= 0 else total_indices
        permuted_tokens_grad = torch.zeros(total_indices, hidden_size, dtype=unpermuted_tokens_grad.dtype, device=unpermuted_tokens_grad.device)
        has_probs = probs is not None and probs.numel() > 0
        if not has_probs:
            for i in range(total_indices):
                idx = sorted_indices[i].item()
                if start <= idx < end:
                    token_idx = i // topk_num
                    permuted_tokens_grad[idx - start] = unpermuted_tokens_grad[token_idx]
            probs_grad = torch.zeros(tokens_num, topk_num, dtype=unpermuted_tokens_grad.dtype, device=unpermuted_tokens_grad.device)
        else:
            topk = probs.shape[1] if probs.dim() == 2 else topk_num
            probs_grad = torch.zeros_like(probs)
            for i in range(total_indices):
                idx = sorted_indices[i].item()
                token_idx = i // topk
                k_idx = i % topk
                if start <= idx < end:
                    permuted_tokens_grad[idx - start] = unpermuted_tokens_grad[token_idx] * probs[token_idx, k_idx]
            if permuted_tokens is not None and permuted_tokens.numel() > 0:
                for i in range(total_indices):
                    idx = sorted_indices[i].item()
                    token_idx = i // topk
                    k_idx = i % topk
                    if start <= idx < end:
                        prod = permuted_tokens[idx - start].float() * unpermuted_tokens_grad[token_idx].float()
                        probs_grad[token_idx, k_idx] = prod.sum().to(probs.dtype)
        return (permuted_tokens_grad, probs_grad)

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level3_34_MoeTokenUnpermuteWithEpGrad.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        unpermuted_tokens_grad_info = inputs[0]
        sorted_indices_info = inputs[1]
        permuted_tokens_info = inputs[2]
        probs_info = inputs[3]
        padded_mode_info = inputs[4]
        restore_shape_info = inputs[5]
        range_vals_info = inputs[6]
        topk_num_info = inputs[7]

        if "data" in unpermuted_tokens_grad_info:
            unpermuted_tokens_grad = torch.tensor(unpermuted_tokens_grad_info["data"], dtype=DTYPE_MAP[unpermuted_tokens_grad_info["dtype"]]).reshape(unpermuted_tokens_grad_info["shape"])
        else:
            unpermuted_tokens_grad = torch.randn(unpermuted_tokens_grad_info["shape"], dtype=DTYPE_MAP[unpermuted_tokens_grad_info["dtype"]])
        if "data" in sorted_indices_info:
            sorted_indices = torch.tensor(sorted_indices_info["data"], dtype=DTYPE_MAP[sorted_indices_info["dtype"]]).reshape(sorted_indices_info["shape"])
        else:
            sorted_indices = torch.randperm(sorted_indices_info["shape"][0], dtype=DTYPE_MAP[sorted_indices_info["dtype"]]) + sorted_indices_info["range"][0]
        if "data" in permuted_tokens_info:
            permuted_tokens = torch.tensor(permuted_tokens_info["data"], dtype=DTYPE_MAP[permuted_tokens_info["dtype"]]).reshape(permuted_tokens_info["shape"])
        else:
            permuted_tokens = torch.randn(permuted_tokens_info["shape"], dtype=DTYPE_MAP[permuted_tokens_info["dtype"]])
        if probs_info["type"] == "attr":
            if probs_info.get("dtype") == "none":
                probs = None
            else:
                probs = probs_info["value"]
        else:
            if "data" in probs_info:
                probs = torch.tensor(probs_info["data"], dtype=DTYPE_MAP[probs_info["dtype"]]).reshape(probs_info["shape"])
            else:
                probs = torch.randn(probs_info["shape"], dtype=DTYPE_MAP[probs_info["dtype"]])
        padded_mode = padded_mode_info["value"]
        restore_shape = restore_shape_info["value"]
        range_vals = range_vals_info["value"]
        topk_num = topk_num_info["value"]

        input_groups.append([unpermuted_tokens_grad, sorted_indices, permuted_tokens, probs, padded_mode, restore_shape, range_vals, topk_num])
    return input_groups


def get_init_inputs():
    return []
