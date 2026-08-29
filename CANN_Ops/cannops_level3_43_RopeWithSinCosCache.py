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

from typing import List, Tuple
import torch
import torch.nn as nn

class Model(nn.Module):
    """PyTorch golden model for RoPE with sin/cos cache."""

    def __init__(self, num_q_heads: int, num_kv_heads: int, head_size: int, is_neox_style: bool=True, mrope_section: list=None, cache_mode: int=0):
        super(Model, self).__init__()
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.is_neox_style = is_neox_style
        self.mrope_section = mrope_section if mrope_section is not None else [0, 0, 0]
        self.cache_mode = cache_mode

    def forward(self, positions: torch.Tensor, query_in: torch.Tensor, key_in: torch.Tensor, cos_sin_cache: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        query_out = self._apply_rope(positions, query_in, cos_sin_cache, self.num_q_heads)
        key_out = self._apply_rope(positions, key_in, cos_sin_cache, self.num_kv_heads)
        return (query_out, key_out)

    def _apply_rope(self, positions: torch.Tensor, x: torch.Tensor, cos_sin_cache: torch.Tensor, num_heads: int) -> torch.Tensor:
        num_tokens = x.shape[0]
        head_size = self.head_size
        rotary_dim = cos_sin_cache.shape[-1]
        orig_dtype = x.dtype
        x_heads = x.reshape(num_tokens, num_heads, head_size).float()
        x_rot = x_heads[:, :, :rotary_dim]
        x_pass = x_heads[:, :, rotary_dim:]
        cos_all = cos_sin_cache[:, :rotary_dim // 2].float()
        sin_all = cos_sin_cache[:, rotary_dim // 2:].float()
        if len(positions.shape) == 1:
            pos = positions.long()
        else:
            pos = positions[0].long()
        cos_vals = cos_all[pos]
        sin_vals = sin_all[pos]
        cos_vals = cos_vals.unsqueeze(1)
        sin_vals = sin_vals.unsqueeze(1)
        if self.is_neox_style:
            x1 = x_rot[:, :, :rotary_dim // 2]
            x2 = x_rot[:, :, rotary_dim // 2:]
            out_rot = torch.cat([x1 * cos_vals - x2 * sin_vals, x2 * cos_vals + x1 * sin_vals], dim=-1)
        else:
            x1 = x_rot[:, :, 0::2]
            x2 = x_rot[:, :, 1::2]
            rotated = torch.cat([x1 * cos_vals - x2 * sin_vals, x2 * cos_vals + x1 * sin_vals], dim=-1)
            out_rot = torch.stack([rotated[:, :, :rotary_dim // 2], rotated[:, :, rotary_dim // 2:]], dim=-1)
            out_rot = out_rot.reshape(num_tokens, num_heads, rotary_dim)
        out_heads = torch.cat([out_rot.to(orig_dtype), x_pass.to(orig_dtype)], dim=-1)
        return out_heads.reshape(num_tokens, num_heads * head_size)

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level3_43_RopeWithSinCosCache.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        positions_info = inputs[0]
        query_in_info = inputs[1]
        key_in_info = inputs[2]
        cos_sin_cache_info = inputs[3]

        if "data" in positions_info:
            positions = torch.tensor(positions_info["data"], dtype=DTYPE_MAP[positions_info["dtype"]]).reshape(positions_info["shape"])
        else:
            positions = torch.randint(positions_info["range"][0], positions_info["range"][1] + 1, tuple(positions_info["shape"]), dtype=DTYPE_MAP[positions_info["dtype"]])
        if "data" in query_in_info:
            query_in = torch.tensor(query_in_info["data"], dtype=DTYPE_MAP[query_in_info["dtype"]]).reshape(query_in_info["shape"])
        else:
            query_in = torch.randn(query_in_info["shape"], dtype=DTYPE_MAP[query_in_info["dtype"]])
        if "data" in key_in_info:
            key_in = torch.tensor(key_in_info["data"], dtype=DTYPE_MAP[key_in_info["dtype"]]).reshape(key_in_info["shape"])
        else:
            key_in = torch.randn(key_in_info["shape"], dtype=DTYPE_MAP[key_in_info["dtype"]])
        if "data" in cos_sin_cache_info:
            cos_sin_cache = torch.tensor(cos_sin_cache_info["data"], dtype=DTYPE_MAP[cos_sin_cache_info["dtype"]]).reshape(cos_sin_cache_info["shape"])
        else:
            cos_sin_cache = torch.randn(cos_sin_cache_info["shape"], dtype=DTYPE_MAP[cos_sin_cache_info["dtype"]])

        input_groups.append([positions, query_in, key_in, cos_sin_cache])
    return input_groups


def get_init_inputs():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level3_43_RopeWithSinCosCache.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    init_groups = []
    for case in cases:
        entries = case.get("init_inputs", [])
        num_q_heads_info = entries[0]
        num_kv_heads_info = entries[1]
        head_size_info = entries[2]
        is_neox_style_info = entries[3]
        mrope_section_info = entries[4]
        cache_mode_info = entries[5]
        num_q_heads = num_q_heads_info["value"]
        num_kv_heads = num_kv_heads_info["value"]
        head_size = head_size_info["value"]
        is_neox_style = is_neox_style_info["value"]
        mrope_section = mrope_section_info["value"]
        cache_mode = cache_mode_info["value"]
        init_groups.append([num_q_heads, num_kv_heads, head_size, is_neox_style, mrope_section, cache_mode])
    return init_groups
