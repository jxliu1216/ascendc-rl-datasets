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

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, log_probs: torch.Tensor, targets: torch.Tensor, input_lengths: List[int], target_lengths: List[int], blank: int, zero_infinity: bool):
        neg_log_likelihood, log_alpha = torch.ops.aten._ctc_loss(log_probs.float(), targets, input_lengths, target_lengths, blank=blank, zero_infinity=zero_infinity)
        res = [neg_log_likelihood.cpu(), log_alpha.cpu()]
        res_list = []
        for i in range(len(res)):
            t = res[i]
            t = torch.where(torch.isinf(t) & (t > 0), torch.tensor(1, dtype=t.dtype, device=t.device), t).to(log_probs.dtype)
            t = torch.where(torch.isinf(t) & (t < 0), torch.tensor(-1, dtype=t.dtype, device=t.device), t).to(log_probs.dtype)
            t = torch.where(torch.isnan(t), torch.tensor(0, dtype=t.dtype, device=t.device), t).to(log_probs.dtype)
            res_list.append(t)
        return res_list

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level2_20_CTCLossV3.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        log_probs_info = inputs[0]
        targets_info = inputs[1]
        input_lengths_info = inputs[2]
        target_lengths_info = inputs[3]
        blank_info = inputs[4]
        zero_infinity_info = inputs[5]

        if "data" in log_probs_info:
            log_probs = torch.tensor(log_probs_info["data"], dtype=DTYPE_MAP[log_probs_info["dtype"]]).reshape(log_probs_info["shape"])
        else:
            log_probs = torch.rand(log_probs_info["shape"], dtype=DTYPE_MAP[log_probs_info["dtype"]]) * (log_probs_info["range"][1] - log_probs_info["range"][0]) + log_probs_info["range"][0]
        if "data" in targets_info:
            targets = torch.tensor(targets_info["data"], dtype=DTYPE_MAP[targets_info["dtype"]]).reshape(targets_info["shape"])
        else:
            targets = torch.randint(targets_info["range"][0], targets_info["range"][1] + 1, tuple(targets_info["shape"]), dtype=DTYPE_MAP[targets_info["dtype"]])
        input_lengths = input_lengths_info["value"]
        target_lengths = target_lengths_info["value"]
        blank = blank_info["value"]
        zero_infinity = zero_infinity_info["value"]

        input_groups.append([log_probs, targets, input_lengths, target_lengths, blank, zero_infinity])
    return input_groups


def get_init_inputs():
    return []
