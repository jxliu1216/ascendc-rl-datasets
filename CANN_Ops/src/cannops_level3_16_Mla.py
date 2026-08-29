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
import math
import torch.nn.functional as F

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()

    def _group_matmul(self, q, k_or_v):
        """Helper for Grouped-Query Attention matmul."""
        num_heads, q_seqlen, _ = q.shape
        kv_heads = k_or_v.shape[0]
        if num_heads == kv_heads:
            return torch.matmul(q, k_or_v)
        group_num = num_heads // kv_heads
        q = q.view(kv_heads, group_num, q_seqlen, -1)
        if k_or_v.dim() == 3:
            k = k_or_v.unsqueeze(1)
            score = torch.matmul(q, k)
        else:
            v = k_or_v.unsqueeze(1)
            score = torch.matmul(q, v)
        return score.view(num_heads, q_seqlen, -1)

    def _ref_masked_attention(self, query, key, value, scale, mask=None):
        """Performs a single scaled dot-product attention operation."""
        q = query.permute(1, 0, 2)
        k = key.permute(1, 2, 0)
        v = value.permute(1, 0, 2)
        scores = self._group_matmul(q, k) * scale
        if mask is not None:
            scores += mask
        attn = F.softmax(scores, dim=-1)
        output = self._group_matmul(attn, v)
        return output.permute(1, 0, 2)

    def forward(self, query_nope, query_rope, kv_nope_cache, kv_rope_cache, block_tables, q_seqlen_list, k_seqlen_list, mask=None):
        """
        Forward pass for the reference Paged Attention model.
        It computes the result in float32 for high precision.
        """
        query = torch.concat([query_nope, query_rope], dim=-1)
        key_cache = torch.concat([kv_nope_cache, kv_rope_cache], dim=-1)
        output_shape = (query.shape[0], query.shape[1], kv_nope_cache.shape[3])
        final_output = torch.empty(output_shape, dtype=torch.float32, device=query_nope.device)
        block_size = kv_nope_cache.shape[1]
        cu_q_seqlen = 0
        for i in range(len(q_seqlen_list)):
            q_len = q_seqlen_list[i]
            k_len = k_seqlen_list[i]
            q_current = query[cu_q_seqlen:cu_q_seqlen + q_len]
            k_list, v_list = ([], [])
            for j in range(k_len):
                block_idx = j // block_size
                block_offset = j % block_size
                block_number = block_tables[i, block_idx].item()
                k_list.append(key_cache[block_number, block_offset])
                v_list.append(kv_nope_cache[block_number, block_offset])
            keys = torch.stack(k_list, dim=0)
            values = torch.stack(v_list, dim=0)
            scale = 1.0 / keys.shape[-1] ** 0.5
            current_mask = mask[cu_q_seqlen:cu_q_seqlen + q_len, :k_len] if mask is not None else None
            out = self._ref_masked_attention(q_current.to(torch.float32), keys.to(torch.float32), values.to(torch.float32), scale, current_mask.to(torch.float32) if current_mask is not None else None)
            final_output[cu_q_seqlen:cu_q_seqlen + q_len] = out
            cu_q_seqlen += q_len
        return final_output

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level3_16_Mla.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        query_nope_info = inputs[0]
        query_rope_info = inputs[1]
        kv_nope_cache_info = inputs[2]
        kv_rope_cache_info = inputs[3]
        block_tables_info = inputs[4]
        q_seqlen_list_info = inputs[5]
        k_seqlen_list_info = inputs[6]
        mask_info = inputs[7]

        if "data" in query_nope_info:
            query_nope = torch.tensor(query_nope_info["data"], dtype=DTYPE_MAP[query_nope_info["dtype"]]).reshape(query_nope_info["shape"])
        else:
            query_nope = torch.rand(query_nope_info["shape"], dtype=DTYPE_MAP[query_nope_info["dtype"]]) * (query_nope_info["range"][1] - query_nope_info["range"][0]) + query_nope_info["range"][0]
        if "data" in query_rope_info:
            query_rope = torch.tensor(query_rope_info["data"], dtype=DTYPE_MAP[query_rope_info["dtype"]]).reshape(query_rope_info["shape"])
        else:
            query_rope = torch.rand(query_rope_info["shape"], dtype=DTYPE_MAP[query_rope_info["dtype"]]) * (query_rope_info["range"][1] - query_rope_info["range"][0]) + query_rope_info["range"][0]
        if "data" in kv_nope_cache_info:
            kv_nope_cache = torch.tensor(kv_nope_cache_info["data"], dtype=DTYPE_MAP[kv_nope_cache_info["dtype"]]).reshape(kv_nope_cache_info["shape"])
        else:
            kv_nope_cache = torch.rand(kv_nope_cache_info["shape"], dtype=DTYPE_MAP[kv_nope_cache_info["dtype"]]) * (kv_nope_cache_info["range"][1] - kv_nope_cache_info["range"][0]) + kv_nope_cache_info["range"][0]
        if "data" in kv_rope_cache_info:
            kv_rope_cache = torch.tensor(kv_rope_cache_info["data"], dtype=DTYPE_MAP[kv_rope_cache_info["dtype"]]).reshape(kv_rope_cache_info["shape"])
        else:
            kv_rope_cache = torch.rand(kv_rope_cache_info["shape"], dtype=DTYPE_MAP[kv_rope_cache_info["dtype"]]) * (kv_rope_cache_info["range"][1] - kv_rope_cache_info["range"][0]) + kv_rope_cache_info["range"][0]
        if "data" in block_tables_info:
            block_tables = torch.tensor(block_tables_info["data"], dtype=DTYPE_MAP[block_tables_info["dtype"]]).reshape(block_tables_info["shape"])
        else:
            block_tables = torch.randint(block_tables_info["range"][0], block_tables_info["range"][1] + 1, tuple(block_tables_info["shape"]), dtype=DTYPE_MAP[block_tables_info["dtype"]])
        q_seqlen_list = q_seqlen_list_info["value"]
        k_seqlen_list = k_seqlen_list_info["value"]
        mask = None

        input_groups.append([query_nope, query_rope, kv_nope_cache, kv_rope_cache, block_tables, q_seqlen_list, k_seqlen_list, mask])
    return input_groups


def get_init_inputs():
    return []
