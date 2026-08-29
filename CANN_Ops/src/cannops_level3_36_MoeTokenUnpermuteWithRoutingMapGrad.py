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

    def forward(self, unpermuted_tokens_grad, out_index, permute_token_id, routing_map, permuted_tokens, probs, drop_and_pad, restore_shape):
        """
        Args:
            unpermuted_tokens_grad: [tokens_num, hidden_size] - gradient from forward output
            out_index: [total_length] - output position indices
            permute_token_id: [total_length] - token id for each position
            routing_map: optional [tokens_num, experts_num] bool/int8 mask
            permuted_tokens: optional [total_length, hidden_size] - forward permuted tokens
            probs: optional [tokens_num, experts_num] - expert probabilities
            drop_and_pad: bool - padded mode flag
            restore_shape: List[int] - shape info for padded mode
        Returns:
            permuted_tokens_grad: [out_length, hidden_size]
            probs_grad: optional [tokens_num, experts_num]
        """
        orig_dtype = unpermuted_tokens_grad.dtype
        tokens_num = unpermuted_tokens_grad.shape[0]
        hidden_size = unpermuted_tokens_grad.shape[1]
        total_length = out_index.shape[0]
        grad_f = unpermuted_tokens_grad.float()
        out_index_cpu = out_index.cpu().long()
        permute_token_id_cpu = permute_token_id.cpu().long()
        if probs is None:
            permuted_tokens_grad = torch.zeros(total_length, hidden_size, dtype=torch.float32, device=unpermuted_tokens_grad.device)
            for i in range(total_length):
                tid = permute_token_id_cpu[i].item()
                oi = out_index_cpu[i].item()
                permuted_tokens_grad[oi] = grad_f[tid]
            return (permuted_tokens_grad.to(orig_dtype), None)
        num_experts = probs.shape[1]
        probs_cpu = probs.float().cpu()
        permuted_tokens_f = permuted_tokens.float()
        permuted_tokens_cpu = permuted_tokens_f.cpu()
        permuted_tokens_grad = torch.zeros(total_length, hidden_size, dtype=torch.float32, device=unpermuted_tokens_grad.device)
        for i in range(total_length):
            tid = permute_token_id_cpu[i].item()
            oi = out_index_cpu[i].item()
            permuted_tokens_grad[oi] = grad_f[tid]
        if not drop_and_pad:
            topK = total_length // tokens_num
            permuted_probs_grad = permuted_tokens_grad * permuted_tokens_f
            probs_grad_expert_order = permuted_probs_grad.sum(dim=-1)
            probs_grad = torch.zeros(tokens_num, num_experts, dtype=torch.float32, device=unpermuted_tokens_grad.device)
            if routing_map is not None:
                routing_map_cpu = routing_map.cpu()
                idx = 0
                for t in range(tokens_num):
                    for e in range(num_experts):
                        if routing_map_cpu[t, e].item():
                            if idx < total_length:
                                probs_grad[t, e] = probs_grad_expert_order[idx].item()
                                idx += 1
            permuted_probs = []
            if routing_map is not None:
                routing_map_cpu = routing_map.cpu()
                for t in range(tokens_num):
                    for e in range(num_experts):
                        if routing_map_cpu[t, e].item():
                            permuted_probs.append(probs_cpu[t, e].item())
            if len(permuted_probs) > 0:
                permuted_probs_tensor = torch.tensor(permuted_probs, dtype=torch.float32, device=unpermuted_tokens_grad.device)
                permuted_tokens_grad = permuted_probs_tensor.unsqueeze(-1) * permuted_tokens_grad
        else:
            capacity = total_length // num_experts
            permuted_probs_grad = permuted_tokens_grad * permuted_tokens_f
            probs_grad_expert_order = permuted_probs_grad.sum(dim=-1)
            probs_grad = torch.zeros(tokens_num, num_experts, dtype=torch.float32, device=unpermuted_tokens_grad.device)
            for i in range(total_length):
                tid = permute_token_id_cpu[i].item()
                oi = out_index_cpu[i].item()
                expert_id = oi // capacity
                probs_grad[tid, expert_id] = probs_grad_expert_order[oi].item()
            for i in range(total_length):
                tid = permute_token_id_cpu[i].item()
                oi = out_index_cpu[i].item()
                expert_id = oi // capacity
                prob_val = probs_cpu[tid, expert_id].item()
                permuted_tokens_grad[oi] = prob_val * permuted_tokens_grad[oi]
        return (permuted_tokens_grad.to(orig_dtype), probs_grad.to(probs.dtype))

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level3_36_MoeTokenUnpermuteWithRoutingMapGrad.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        unpermuted_tokens_grad_info = inputs[0]
        out_index_info = inputs[1]
        permute_token_id_info = inputs[2]
        routing_map_info = inputs[3]
        permuted_tokens_info = inputs[4]
        probs_info = inputs[5]
        drop_and_pad_info = inputs[6]
        restore_shape_info = inputs[7]

        if "data" in unpermuted_tokens_grad_info:
            unpermuted_tokens_grad = torch.tensor(unpermuted_tokens_grad_info["data"], dtype=DTYPE_MAP[unpermuted_tokens_grad_info["dtype"]]).reshape(unpermuted_tokens_grad_info["shape"])
        else:
            unpermuted_tokens_grad = torch.randn(unpermuted_tokens_grad_info["shape"], dtype=DTYPE_MAP[unpermuted_tokens_grad_info["dtype"]]) * unpermuted_tokens_grad_info["std"] + unpermuted_tokens_grad_info["mean"]
        if "data" in out_index_info:
            out_index = torch.tensor(out_index_info["data"], dtype=DTYPE_MAP[out_index_info["dtype"]]).reshape(out_index_info["shape"])
        else:
            out_index = torch.arange(out_index_info["range"][0], out_index_info["range"][0] + out_index_info["shape"][0], dtype=DTYPE_MAP[out_index_info["dtype"]]).reshape(out_index_info["shape"])
        if "data" in permute_token_id_info:
            permute_token_id = torch.tensor(permute_token_id_info["data"], dtype=DTYPE_MAP[permute_token_id_info["dtype"]]).reshape(permute_token_id_info["shape"])
        else:
            permute_token_id = torch.randint(permute_token_id_info["range"][0], permute_token_id_info["range"][1] + 1, tuple(permute_token_id_info["shape"]), dtype=DTYPE_MAP[permute_token_id_info["dtype"]])
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
        if permuted_tokens_info["type"] == "attr":
            if permuted_tokens_info.get("dtype") == "none":
                permuted_tokens = None
            else:
                permuted_tokens = permuted_tokens_info["value"]
        else:
            if "data" in permuted_tokens_info:
                permuted_tokens = torch.tensor(permuted_tokens_info["data"], dtype=DTYPE_MAP[permuted_tokens_info["dtype"]]).reshape(permuted_tokens_info["shape"])
            else:
                permuted_tokens = torch.randn(permuted_tokens_info["shape"], dtype=DTYPE_MAP[permuted_tokens_info["dtype"]]) * permuted_tokens_info["std"] + permuted_tokens_info["mean"]
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

        input_groups.append([unpermuted_tokens_grad, out_index, permute_token_id, routing_map, permuted_tokens, probs, drop_and_pad, restore_shape])
    return input_groups


def get_init_inputs():
    return []
