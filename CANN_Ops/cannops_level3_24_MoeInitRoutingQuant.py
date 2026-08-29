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
import numpy as np

def moe_init_routing(x, row_idx, expert_idx, active_num):
    num_rows = x.shape[0]
    hidden_size = x.shape[-1]
    k = expert_idx.shape[-1]
    sort_expert_for_source_row = np.argsort(expert_idx.reshape((-1,)), axis=-1, kind='stable')
    expanded_expert_idx = np.sort(expert_idx.reshape((-1,)), axis=-1)
    expanded_dst_to_src_row = np.take_along_axis(row_idx.reshape((-1,)), sort_expert_for_source_row, axis=-1)
    expanded_row_idx = np.zeros(expanded_dst_to_src_row.shape).astype(np.int32)
    expanded_row_idx[expanded_dst_to_src_row] = np.arange(expanded_dst_to_src_row.shape[-1])
    active_num = min(active_num, num_rows) * k
    expanded_x = x[expanded_dst_to_src_row[:active_num] % num_rows, :]
    return (expanded_x, expanded_row_idx, expanded_expert_idx)

def quant_a(x, scale, offset):
    sr = x.astype(np.float16) * scale + offset
    roundd = np.rint(sr)
    roundd = np.clip(roundd, -128, 127)
    output = roundd.astype(np.int8)
    return output

class Model(torch.nn.Module):

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, row_idx, expert_idx, active_num, scale, offset):
        expanded_x, expanded_row_idx, expanded_expert_idx = moe_init_routing(x.to(torch.float32).numpy(), row_idx.numpy(), expert_idx.numpy(), active_num)
        expanded_x = quant_a(expanded_x, scale, offset)
        return (torch.from_numpy(expanded_x), torch.from_numpy(expanded_row_idx.astype(np.int32)), torch.from_numpy(expanded_expert_idx))

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level3_24_MoeInitRoutingQuant.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        x_info = inputs[0]
        row_idx_info = inputs[1]
        expert_idx_info = inputs[2]
        active_num_info = inputs[3]
        scale_info = inputs[4]
        offset_info = inputs[5]

        if "data" in x_info:
            x = torch.tensor(x_info["data"], dtype=DTYPE_MAP[x_info["dtype"]]).reshape(x_info["shape"])
        else:
            x = torch.randn(x_info["shape"], dtype=DTYPE_MAP[x_info["dtype"]])
        if "data" in row_idx_info:
            row_idx = torch.tensor(row_idx_info["data"], dtype=DTYPE_MAP[row_idx_info["dtype"]]).reshape(row_idx_info["shape"])
        else:
            row_idx = torch.randint(row_idx_info["range"][0], row_idx_info["range"][1] + 1, tuple(row_idx_info["shape"]), dtype=DTYPE_MAP[row_idx_info["dtype"]])
        if "data" in expert_idx_info:
            expert_idx = torch.tensor(expert_idx_info["data"], dtype=DTYPE_MAP[expert_idx_info["dtype"]]).reshape(expert_idx_info["shape"])
        else:
            expert_idx = torch.full(expert_idx_info["shape"], expert_idx_info["fill"], dtype=DTYPE_MAP[expert_idx_info["dtype"]])
        active_num = active_num_info["value"]
        scale = scale_info["value"]
        offset = offset_info["value"]

        input_groups.append([x, row_idx, expert_idx, active_num, scale, offset])
    return input_groups


def get_init_inputs():
    return []
