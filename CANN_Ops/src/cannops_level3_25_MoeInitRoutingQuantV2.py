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

def adapter_capacity(sorted_row_idx, sorted_expert_idx, capacity):
    count = 0
    last = sorted_expert_idx[0]
    for i, val in enumerate(sorted_expert_idx):
        if last != val:
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

    def forward(self, x, expert_idx, scale_optional, offset_optional, active_num, expert_capacity, expert_num, drop_pad_mode, expert_tokens_count_or_cumsum_flag, expert_tokens_before_capacity_flag, quant_mode):
        input_x = x.to(torch.float32).numpy()
        expert_idx = expert_idx.numpy()
        scale = scale_optional.numpy()
        offset_t = offset_optional.numpy()
        num_rows = input_x.shape[0]
        hidden_size = input_x.shape[-1]
        k = expert_idx.shape[-1]
        flat_expert = expert_idx.reshape(-1)
        sorted_row_idx = np.argsort(flat_expert, kind='stable')
        sorted_expert_idx = flat_expert[sorted_row_idx].astype(np.int32, copy=True)
        if drop_pad_mode == 1 and expert_num <= 0:
            return
        expert_tokens_count_or_cumsum = None
        expert_tokens_before_capacity = None
        expert_idx_hist = np.bincount(sorted_expert_idx, minlength=expert_num).astype(np.int64)
        expert_token_idx = np.cumsum(expert_idx_hist)
        if drop_pad_mode == 1 and expert_tokens_before_capacity_flag:
            expert_tokens_before_capacity = expert_idx_hist.astype('int32')
        if drop_pad_mode == 0 and expert_tokens_count_or_cumsum_flag == 1:
            expert_tokens_count_or_cumsum = expert_token_idx.astype('int32')
        elif drop_pad_mode == 0 and expert_tokens_count_or_cumsum_flag == 2:
            expert_tokens_count_or_cumsum = expert_idx_hist.astype('int32')
        slot_filled = None
        if drop_pad_mode == 0:
            expanded_row_idx = np.zeros(sorted_row_idx.shape, dtype=np.int32)
            expanded_row_idx[sorted_row_idx] = np.arange(sorted_row_idx.shape[-1], dtype=np.int32)
            if active_num == 0:
                active_num = num_rows * k
            else:
                active_num = min(active_num, num_rows * k)
            expanded_x = input_x[sorted_row_idx[:active_num] // k, :]
        else:
            adapter_capacity(sorted_row_idx, sorted_expert_idx, expert_capacity)
            sort_row_tmp = np.full(expert_num * expert_capacity, -1, dtype=int)
            offset = 0
            lastExpertId = 0
            for i, val in enumerate(sorted_row_idx):
                if val != -1:
                    if lastExpertId != sorted_expert_idx[i]:
                        offset = 0
                        lastExpertId = sorted_expert_idx[i]
                    sort_row_tmp[sorted_expert_idx[i] * expert_capacity + offset] = sorted_row_idx[i]
                    offset += 1
            expanded_row_idx = np.full(sorted_row_idx.shape, -1)
            for i, val in enumerate(sort_row_tmp):
                if val != -1:
                    expanded_row_idx[val] = i
            expanded_x = np.full((expert_num * expert_capacity, hidden_size), 0, dtype=input_x.dtype)
            slot_filled_flat = np.zeros(expert_num * expert_capacity, dtype=bool)
            for i, val in enumerate(sort_row_tmp):
                if val != -1:
                    expanded_x[i] = input_x[val // k]
                    slot_filled_flat[i] = True
            expanded_x = expanded_x.reshape(expert_num, expert_capacity, hidden_size)
            slot_filled = slot_filled_flat.reshape(expert_num, expert_capacity)
        if expert_tokens_count_or_cumsum is None:
            expert_tokens_count_or_cumsum = torch.tensor([])
        else:
            expert_tokens_count_or_cumsum = torch.from_numpy(expert_tokens_count_or_cumsum)
        ds = torch.tensor([])
        if quant_mode == 0:
            expanded_x = np.clip(expanded_x, np.finfo(np.float16).min, np.finfo(np.float16).max)
            expanded_x = expanded_x.astype(np.float16)
            scale_v = np.clip(scale[0], np.finfo(np.float16).min, np.finfo(np.float16).max)
            offset_v = offset_t.astype('float16')
            rr = expanded_x * scale_v + offset_v
            roundd = np.rint(rr)
            roundd = np.clip(roundd, -128, 127)
            roundd = roundd.astype('int8')
            if slot_filled is not None:
                mask = slot_filled[:, :, np.newaxis]
                expanded_x = np.where(mask, roundd, np.int8(0))
            else:
                expanded_x = roundd
        else:
            xf = expanded_x.astype('float32')
            xa = np.abs(xf)
            xm = np.max(xa, axis=-1, keepdims=True)
            ds_arr = xm / 127.0
            q = np.round(np.where(xm > 0, xf / np.maximum(ds_arr, 1e-30), 0.0)).astype('int8')
            if slot_filled is not None:
                mask = slot_filled[:, :, np.newaxis]
                expanded_x = np.where(mask, q, np.int8(0))
            else:
                expanded_x = q
            ds = torch.from_numpy(ds_arr).reshape(-1)
        t_expanded_x = torch.from_numpy(expanded_x)
        if expert_tokens_before_capacity is None:
            expert_tokens_before_capacity = torch.tensor([])
        else:
            expert_tokens_before_capacity = torch.from_numpy(expert_tokens_before_capacity)
        return (t_expanded_x, torch.from_numpy(expanded_row_idx.astype('int32')), expert_tokens_count_or_cumsum, expert_tokens_before_capacity, ds)

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level3_25_MoeInitRoutingQuantV2.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        x_info = inputs[0]
        expert_idx_info = inputs[1]
        scale_optional_info = inputs[2]
        offset_optional_info = inputs[3]
        active_num_info = inputs[4]
        expert_capacity_info = inputs[5]
        expert_num_info = inputs[6]
        drop_pad_mode_info = inputs[7]
        expert_tokens_count_or_cumsum_flag_info = inputs[8]
        expert_tokens_before_capacity_flag_info = inputs[9]
        quant_mode_info = inputs[10]

        if "data" in x_info:
            x = torch.tensor(x_info["data"], dtype=DTYPE_MAP[x_info["dtype"]]).reshape(x_info["shape"])
        else:
            x = torch.randn(x_info["shape"], dtype=DTYPE_MAP[x_info["dtype"]])
        if "data" in expert_idx_info:
            expert_idx = torch.tensor(expert_idx_info["data"], dtype=DTYPE_MAP[expert_idx_info["dtype"]]).reshape(expert_idx_info["shape"])
        else:
            expert_idx = torch.full(expert_idx_info["shape"], expert_idx_info["fill"], dtype=DTYPE_MAP[expert_idx_info["dtype"]])
        if "data" in scale_optional_info:
            scale_optional = torch.tensor(scale_optional_info["data"], dtype=DTYPE_MAP[scale_optional_info["dtype"]]).reshape(scale_optional_info["shape"])
        else:
            scale_optional = torch.full(scale_optional_info["shape"], scale_optional_info["fill"], dtype=DTYPE_MAP[scale_optional_info["dtype"]])
        if "data" in offset_optional_info:
            offset_optional = torch.tensor(offset_optional_info["data"], dtype=DTYPE_MAP[offset_optional_info["dtype"]]).reshape(offset_optional_info["shape"])
        else:
            offset_optional = torch.full(offset_optional_info["shape"], offset_optional_info["fill"], dtype=DTYPE_MAP[offset_optional_info["dtype"]])
        active_num = active_num_info["value"]
        expert_capacity = expert_capacity_info["value"]
        expert_num = expert_num_info["value"]
        drop_pad_mode = drop_pad_mode_info["value"]
        expert_tokens_count_or_cumsum_flag = expert_tokens_count_or_cumsum_flag_info["value"]
        expert_tokens_before_capacity_flag = expert_tokens_before_capacity_flag_info["value"]
        quant_mode = quant_mode_info["value"]

        input_groups.append([x, expert_idx, scale_optional, offset_optional, active_num, expert_capacity, expert_num, drop_pad_mode, expert_tokens_count_or_cumsum_flag, expert_tokens_before_capacity_flag, quant_mode])
    return input_groups


def get_init_inputs():
    return []
