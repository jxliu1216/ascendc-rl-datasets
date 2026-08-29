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

from typing import Any
import torch
import numpy

def adapter_capacity(sorted_row_idx, sorted_expert_idx, capacity):
    count = 0
    last = sorted_expert_idx[0]
    for i, val in enumerate(sorted_expert_idx):
        if val != last:
            count = 1
            last = val
        else:
            count += 1
            if count > capacity:
                sorted_expert_idx[i] = -1
                sorted_row_idx[i] = -1

class Model(torch.nn.Module):

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, expert_idx, active_num, expert_capacity, expert_num, drop_pad_mode, expert_tokens_count_or_cumsum_flag, expert_tokens_before_capacity_flag):
        x_dtype = x.dtype
        x = x.to(torch.float32).numpy()
        expert_idx = expert_idx.numpy()
        num_rows = x.shape[0]
        hidden_size = x.shape[-1]
        k = expert_idx.shape[-1]
        sorted_row_idx = numpy.argsort(expert_idx.reshape((-1,)), axis=-1, kind='stable')
        sorted_expert_idx = numpy.sort(expert_idx.reshape((-1,)), axis=-1)
        if drop_pad_mode == 1 and expert_num <= 0:
            return
        expert_tokens_count_or_cumsum = None
        expert_tokens_before_capacity = None
        expert_idx_hist, bins = numpy.histogram(sorted_expert_idx, bins=expert_num, range=(0, expert_num - 1))
        expert_token_idx = numpy.cumsum(expert_idx_hist)
        if drop_pad_mode == 1 and expert_tokens_before_capacity_flag:
            expert_tokens_before_capacity = expert_idx_hist.astype('int32')
        if drop_pad_mode == 0 and expert_tokens_count_or_cumsum_flag == 1:
            expert_tokens_count_or_cumsum = expert_token_idx.astype('int32')
        elif drop_pad_mode == 0 and expert_tokens_count_or_cumsum_flag == 2:
            expert_tokens_count_or_cumsum = expert_idx_hist.astype('int32')
        if drop_pad_mode == 0:
            expanded_row_idx = numpy.zeros(sorted_row_idx.shape, dtype=numpy.int32)
            expanded_row_idx[sorted_row_idx] = numpy.arange(sorted_row_idx.shape[-1], dtype=numpy.int32)
            if active_num == 0:
                active_num = num_rows * k
            else:
                active_num = min(active_num, num_rows * k)
            expanded_x = x[sorted_row_idx[:active_num] // k, :]
        else:
            adapter_capacity(sorted_row_idx, sorted_expert_idx, expert_capacity)
            sorted_row_tmp = numpy.full(expert_num * expert_capacity, -1, dtype=int)
            offset = 0
            lastExpertId = 0
            for i, val in enumerate(sorted_row_idx):
                if val != -1:
                    if lastExpertId != sorted_expert_idx[i]:
                        offset = 0
                        lastExpertId = sorted_expert_idx[i]
                    sorted_row_tmp[sorted_expert_idx[i] * expert_capacity + offset] = sorted_row_idx[i]
                    offset = offset + 1
            expanded_row_idx = numpy.full(sorted_row_idx.shape, -1)
            for i, val in enumerate(sorted_row_tmp):
                if val != -1:
                    expanded_row_idx[val] = i
            expanded_x = numpy.full((expert_num * expert_capacity, hidden_size), 0, dtype=x.dtype)
            for i, val in enumerate(sorted_row_tmp):
                if val != -1:
                    expanded_x[i] = x[val // k]
            expanded_x = expanded_x.reshape(expert_num, expert_capacity, hidden_size)
        if expert_tokens_count_or_cumsum is not None:
            expert_tokens_count_or_cumsum = torch.from_numpy(expert_tokens_count_or_cumsum)
        return (torch.from_numpy(expanded_x).to(x_dtype), torch.from_numpy(expanded_row_idx.astype('int32')), expert_tokens_count_or_cumsum, torch.from_numpy(expert_tokens_before_capacity))

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level3_26_MoeInitRoutingV2.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        x_info = inputs[0]
        expert_idx_info = inputs[1]
        active_num_info = inputs[2]
        expert_capacity_info = inputs[3]
        expert_num_info = inputs[4]
        drop_pad_mode_info = inputs[5]
        expert_tokens_count_or_cumsum_flag_info = inputs[6]
        expert_tokens_before_capacity_flag_info = inputs[7]

        if "data" in x_info:
            x = torch.tensor(x_info["data"], dtype=DTYPE_MAP[x_info["dtype"]]).reshape(x_info["shape"])
        else:
            x = torch.randn(x_info["shape"], dtype=DTYPE_MAP[x_info["dtype"]])
        if "data" in expert_idx_info:
            expert_idx = torch.tensor(expert_idx_info["data"], dtype=DTYPE_MAP[expert_idx_info["dtype"]]).reshape(expert_idx_info["shape"])
        else:
            expert_idx = torch.randint(expert_idx_info["range"][0], expert_idx_info["range"][1] + 1, tuple(expert_idx_info["shape"]), dtype=DTYPE_MAP[expert_idx_info["dtype"]])
        active_num = active_num_info["value"]
        expert_capacity = expert_capacity_info["value"]
        expert_num = expert_num_info["value"]
        drop_pad_mode = drop_pad_mode_info["value"]
        expert_tokens_count_or_cumsum_flag = expert_tokens_count_or_cumsum_flag_info["value"]
        expert_tokens_before_capacity_flag = expert_tokens_before_capacity_flag_info["value"]

        input_groups.append([x, expert_idx, active_num, expert_capacity, expert_num, drop_pad_mode, expert_tokens_count_or_cumsum_flag, expert_tokens_before_capacity_flag])
    return input_groups


def get_init_inputs():
    return []
