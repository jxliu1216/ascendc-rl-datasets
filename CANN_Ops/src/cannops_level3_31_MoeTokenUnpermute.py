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
    """参考模型：与 api_desc 一致 — sorted_indices[k] 为从 permuted_tokens 取行的下标（gather），再按 topK 聚合。"""

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, permuted_tokens, sorted_indices, probs=None, padded_mode=False):
        rows, hidden = permuted_tokens.shape
        if probs is not None:
            tokens_num, topk = probs.shape
        else:
            tokens_num = rows
            topk = 1
        if rows != tokens_num * topk:
            raise ValueError(f'permuted_tokens dim0 ({rows}) must equal tokens_num*topk ({tokens_num}*{topk})')
        if sorted_indices.numel() != rows:
            raise ValueError('sorted_indices length must match permuted_tokens rows')
        pt = permuted_tokens.float()
        idx = sorted_indices.to(torch.int64).clamp(0, rows - 1)
        gathered = pt.index_select(0, idx.reshape(-1))
        if probs is not None:
            gathered = gathered * probs.float().reshape(-1, 1)
        out = gathered.view(tokens_num, topk, hidden).sum(dim=1)
        return out.to(permuted_tokens.dtype)

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level3_31_MoeTokenUnpermute.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        permuted_tokens_info = inputs[0]
        sorted_indices_info = inputs[1]
        probs_info = inputs[2]
        padded_mode_info = inputs[3]

        if "data" in permuted_tokens_info:
            permuted_tokens = torch.tensor(permuted_tokens_info["data"], dtype=DTYPE_MAP[permuted_tokens_info["dtype"]]).reshape(permuted_tokens_info["shape"])
        else:
            permuted_tokens = torch.randn(permuted_tokens_info["shape"], dtype=DTYPE_MAP[permuted_tokens_info["dtype"]])
        if "data" in sorted_indices_info:
            sorted_indices = torch.tensor(sorted_indices_info["data"], dtype=DTYPE_MAP[sorted_indices_info["dtype"]]).reshape(sorted_indices_info["shape"])
        else:
            sorted_indices = torch.randperm(sorted_indices_info["shape"][0], dtype=DTYPE_MAP[sorted_indices_info["dtype"]]) + sorted_indices_info["range"][0]
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
        padded_mode = padded_mode_info["value"]

        input_groups.append([permuted_tokens, sorted_indices, probs, padded_mode])
    return input_groups


def get_init_inputs():
    return []
