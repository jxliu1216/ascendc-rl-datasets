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
import torch.nn.functional as F

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, seq_lengths, seq_dim=1, batch_dim=0):
        input_shape = x.shape
        output = torch.zeros_like(x)
        batch_size = input_shape[batch_dim]
        for i in range(batch_size):
            batch_selector = [slice(None)] * len(input_shape)
            batch_selector[batch_dim] = i
            batch_selector = tuple(batch_selector)
            seq_len = seq_lengths[i].item() if seq_lengths.ndim > 0 else seq_lengths
            reversed_indices = torch.arange(seq_len - 1, -1, -1, device=x.device)
            seq_indices = torch.arange(seq_len, device=x.device)
            selector = list(batch_selector)
            selector[seq_dim] = seq_indices
            selector = tuple(selector)
            reversed_selector = list(batch_selector)
            reversed_selector[seq_dim] = reversed_indices
            reversed_selector = tuple(reversed_selector)
            output[selector] = x[reversed_selector]
            if seq_len < input_shape[seq_dim]:
                remaining_selector = list(batch_selector)
                remaining_selector[seq_dim] = slice(seq_len, None)
                remaining_selector = tuple(remaining_selector)
                output[remaining_selector] = x[remaining_selector]
        return output

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level2_137_ReverseSequence.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        x_info = inputs[0]
        seq_lengths_info = inputs[1]
        seq_dim_info = inputs[2]
        batch_dim_info = inputs[3]

        if "data" in x_info:
            x = torch.tensor(x_info["data"], dtype=DTYPE_MAP[x_info["dtype"]]).reshape(x_info["shape"])
        else:
            _dt = DTYPE_MAP[x_info["dtype"]]
            if _dt in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
                x = torch.randint(x_info["range"][0], x_info["range"][1] + 1, tuple(x_info["shape"]), dtype=_dt)
            elif _dt == torch.bool:
                x = torch.rand(x_info["shape"]) < x_info.get("true_frac", 0.5)
            else:
                x = torch.randn(x_info["shape"], dtype=_dt)
        if "data" in seq_lengths_info:
            seq_lengths = torch.tensor(seq_lengths_info["data"], dtype=DTYPE_MAP[seq_lengths_info["dtype"]]).reshape(seq_lengths_info["shape"])
        else:
            seq_lengths = torch.randint(seq_lengths_info["range"][0], seq_lengths_info["range"][1] + 1, tuple(seq_lengths_info["shape"]), dtype=DTYPE_MAP[seq_lengths_info["dtype"]])
        seq_dim = seq_dim_info["value"]
        batch_dim = batch_dim_info["value"]

        input_groups.append([x, seq_lengths, seq_dim, batch_dim])
    return input_groups


def get_init_inputs():
    return []
