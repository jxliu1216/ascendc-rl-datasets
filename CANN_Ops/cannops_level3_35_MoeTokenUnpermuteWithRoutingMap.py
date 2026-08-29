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

    def forward(self, permuted_tokens, sorted_indices, routing_map, probs, drop_and_pad, restore_shape):
        """
        Args:
            permuted_tokens: [num_out_tokens, hidden_size]
            sorted_indices: [total_length] - indices into permuted_tokens
            routing_map: optional [num_tokens, num_experts] bool/int8 mask
            probs: optional [num_tokens, num_experts] - expert probabilities
            drop_and_pad: bool - padded mode flag
            restore_shape: List[int] - [num_tokens] used when probs is None
        Returns:
            unpermuted_tokens: [num_tokens, hidden_size]
        """
        orig_dtype = permuted_tokens.dtype
        num_out_tokens = permuted_tokens.shape[0]
        hidden_size = permuted_tokens.shape[1]
        total_length = sorted_indices.shape[0]
        if probs is not None:
            num_tokens = probs.shape[0]
            num_experts = probs.shape[1]
            topK = num_out_tokens // num_tokens
        else:
            num_tokens = restore_shape[0] if len(restore_shape) > 0 else total_length
            topK = num_out_tokens // num_tokens
        tokens_f = permuted_tokens.float()
        output = torch.zeros(num_tokens, hidden_size, dtype=torch.float32, device=permuted_tokens.device)
        indices_cpu = sorted_indices.cpu()
        tokens_cpu = tokens_f.cpu()
        if probs is not None:
            probs_cpu = probs.float().cpu()
            if routing_map is not None:
                routing_map_cpu = routing_map.cpu()
        for i in range(num_tokens):
            for j in range(topK):
                idx = indices_cpu[i * topK + j].item()
                if idx < num_out_tokens:
                    token = tokens_cpu[idx]
                    if probs is not None:
                        if routing_map is not None:
                            if not routing_map_cpu[i, j].item():
                                continue
                        prob_val = probs_cpu[i, j].item()
                        if prob_val == 0:
                            continue
                        token = token * prob_val
                    output[i] += token
        return output.to(orig_dtype)

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level3_35_MoeTokenUnpermuteWithRoutingMap.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        permuted_tokens_info = inputs[0]
        sorted_indices_info = inputs[1]
        routing_map_info = inputs[2]
        probs_info = inputs[3]
        drop_and_pad_info = inputs[4]
        restore_shape_info = inputs[5]

        if "data" in permuted_tokens_info:
            permuted_tokens = torch.tensor(permuted_tokens_info["data"], dtype=DTYPE_MAP[permuted_tokens_info["dtype"]]).reshape(permuted_tokens_info["shape"])
        else:
            permuted_tokens = torch.randn(permuted_tokens_info["shape"], dtype=DTYPE_MAP[permuted_tokens_info["dtype"]]) * permuted_tokens_info["std"] + permuted_tokens_info["mean"]
        if "data" in sorted_indices_info:
            sorted_indices = torch.tensor(sorted_indices_info["data"], dtype=DTYPE_MAP[sorted_indices_info["dtype"]]).reshape(sorted_indices_info["shape"])
        else:
            sorted_indices = torch.randint(sorted_indices_info["range"][0], sorted_indices_info["range"][1] + 1, tuple(sorted_indices_info["shape"]), dtype=DTYPE_MAP[sorted_indices_info["dtype"]])
        if routing_map_info["type"] == "attr":
            if routing_map_info.get("dtype") == "none":
                routing_map = None
            else:
                routing_map = routing_map_info["value"]
        else:
            if "data" in routing_map_info:
                routing_map = torch.tensor(routing_map_info["data"], dtype=DTYPE_MAP[routing_map_info["dtype"]]).reshape(routing_map_info["shape"])
            else:
                routing_map = torch.full(routing_map_info["shape"], routing_map_info["value"], dtype=torch.bool)
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
        drop_and_pad = drop_and_pad_info["value"]
        restore_shape = restore_shape_info["value"]

        input_groups.append([permuted_tokens, sorted_indices, routing_map, probs, drop_and_pad, restore_shape])
    return input_groups


def get_init_inputs():
    return []
