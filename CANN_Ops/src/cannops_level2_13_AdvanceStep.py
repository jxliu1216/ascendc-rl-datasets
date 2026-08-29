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

from typing import List
import torch
import torch.nn as nn

class Model(nn.Module):
    """AdvanceStep 算子的 PyTorch 参考实现（golden model）。"""

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input_tokens: torch.Tensor, sampled_token_ids: torch.Tensor, input_positions: torch.Tensor, seq_lens: torch.Tensor, slot_mapping: torch.Tensor, block_tables: torch.Tensor, num_seqs: int, num_queries: int, block_size: int) -> List[torch.Tensor]:
        input_tokens = input_tokens.clone()
        input_positions = input_positions.clone()
        seq_lens = seq_lens.clone()
        slot_mapping = slot_mapping.clone()
        n_pad = num_seqs - num_queries
        total_core_num = 48
        if n_pad > 0:
            for i in range(0, n_pad, total_core_num):
                input_tokens[num_queries + i] = 0
                input_positions[num_queries + i] = 0
                slot_mapping[num_queries + i] = -1
        for index in range(num_queries):
            input_tokens[index] = sampled_token_ids[index]
            seq_len = seq_lens[index].item()
            next_seq_len = seq_len + 1
            next_input_pos = next_seq_len - 1
            seq_lens[index] = next_seq_len
            input_positions[index] = next_input_pos
            block_index = next_input_pos // block_size
            block_offset = next_input_pos % block_size
            block_tables_flat = block_tables.flatten()
            slot_num = (block_tables_flat[block_index].item() + block_tables.shape[1] * index) * block_size + block_offset
            slot_mapping[index] = slot_num
        return [input_tokens, input_positions, seq_lens, slot_mapping]

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level2_13_AdvanceStep.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        input_tokens_info = inputs[0]
        sampled_token_ids_info = inputs[1]
        input_positions_info = inputs[2]
        seq_lens_info = inputs[3]
        slot_mapping_info = inputs[4]
        block_tables_info = inputs[5]
        num_seqs_info = inputs[6]
        num_queries_info = inputs[7]
        block_size_info = inputs[8]

        if "data" in input_tokens_info:
            input_tokens = torch.tensor(input_tokens_info["data"], dtype=DTYPE_MAP[input_tokens_info["dtype"]]).reshape(input_tokens_info["shape"])
        else:
            input_tokens = torch.randint(input_tokens_info["range"][0], input_tokens_info["range"][1] + 1, tuple(input_tokens_info["shape"]), dtype=DTYPE_MAP[input_tokens_info["dtype"]])
        if "data" in sampled_token_ids_info:
            sampled_token_ids = torch.tensor(sampled_token_ids_info["data"], dtype=DTYPE_MAP[sampled_token_ids_info["dtype"]]).reshape(sampled_token_ids_info["shape"])
        else:
            sampled_token_ids = torch.randint(sampled_token_ids_info["range"][0], sampled_token_ids_info["range"][1] + 1, tuple(sampled_token_ids_info["shape"]), dtype=DTYPE_MAP[sampled_token_ids_info["dtype"]])
        if "data" in input_positions_info:
            input_positions = torch.tensor(input_positions_info["data"], dtype=DTYPE_MAP[input_positions_info["dtype"]]).reshape(input_positions_info["shape"])
        else:
            input_positions = torch.randint(input_positions_info["range"][0], input_positions_info["range"][1] + 1, tuple(input_positions_info["shape"]), dtype=DTYPE_MAP[input_positions_info["dtype"]])
        if "data" in seq_lens_info:
            seq_lens = torch.tensor(seq_lens_info["data"], dtype=DTYPE_MAP[seq_lens_info["dtype"]]).reshape(seq_lens_info["shape"])
        else:
            seq_lens = torch.randint(seq_lens_info["range"][0], seq_lens_info["range"][1] + 1, tuple(seq_lens_info["shape"]), dtype=DTYPE_MAP[seq_lens_info["dtype"]])
        if "data" in slot_mapping_info:
            slot_mapping = torch.tensor(slot_mapping_info["data"], dtype=DTYPE_MAP[slot_mapping_info["dtype"]]).reshape(slot_mapping_info["shape"])
        else:
            slot_mapping = torch.full(slot_mapping_info["shape"], slot_mapping_info["fill"], dtype=DTYPE_MAP[slot_mapping_info["dtype"]])
        if "data" in block_tables_info:
            block_tables = torch.tensor(block_tables_info["data"], dtype=DTYPE_MAP[block_tables_info["dtype"]]).reshape(block_tables_info["shape"])
        else:
            block_tables = torch.randint(block_tables_info["range"][0], block_tables_info["range"][1] + 1, tuple(block_tables_info["shape"]), dtype=DTYPE_MAP[block_tables_info["dtype"]])
        num_seqs = num_seqs_info["value"]
        num_queries = num_queries_info["value"]
        block_size = block_size_info["value"]

        input_groups.append([input_tokens, sampled_token_ids, input_positions, seq_lens, slot_mapping, block_tables, num_seqs, num_queries, block_size])
    return input_groups


def get_init_inputs():
    return []
