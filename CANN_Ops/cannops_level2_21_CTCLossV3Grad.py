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

    def forward(self, grad: torch.Tensor, log_probs: torch.Tensor, targets: torch.Tensor, input_lengths: List[int], target_lengths: List[int], neg_log_likelihood: torch.Tensor, log_aplha: torch.Tensor, blank: int, zero_infinity: bool):
        res = torch.ops.aten._ctc_loss_backward(grad.float(), log_probs.float(), targets.type(torch.int64), input_lengths, target_lengths, neg_log_likelihood.float(), log_aplha.float(), blank, zero_infinity)
        return res.to(grad.dtype)

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level2_21_CTCLossV3Grad.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        grad_info = inputs[0]
        log_probs_info = inputs[1]
        targets_info = inputs[2]
        input_lengths_info = inputs[3]
        target_lengths_info = inputs[4]
        neg_log_likelihood_info = inputs[5]
        log_aplha_info = inputs[6]
        blank_info = inputs[7]
        zero_infinity_info = inputs[8]

        if "data" in grad_info:
            grad = torch.tensor(grad_info["data"], dtype=DTYPE_MAP[grad_info["dtype"]]).reshape(grad_info["shape"])
        else:
            grad = torch.rand(grad_info["shape"], dtype=DTYPE_MAP[grad_info["dtype"]]) * (grad_info["range"][1] - grad_info["range"][0]) + grad_info["range"][0]
        if "data" in log_probs_info:
            log_probs = torch.tensor(log_probs_info["data"], dtype=DTYPE_MAP[log_probs_info["dtype"]]).reshape(log_probs_info["shape"])
        else:
            log_probs = torch.randn(log_probs_info["shape"], dtype=DTYPE_MAP[log_probs_info["dtype"]]) * log_probs_info["std"] + log_probs_info["mean"]
        if "data" in targets_info:
            targets = torch.tensor(targets_info["data"], dtype=DTYPE_MAP[targets_info["dtype"]]).reshape(targets_info["shape"])
        else:
            targets = torch.randint(targets_info["range"][0], targets_info["range"][1] + 1, tuple(targets_info["shape"]), dtype=DTYPE_MAP[targets_info["dtype"]])
        input_lengths = input_lengths_info["value"]
        target_lengths = target_lengths_info["value"]
        if "data" in neg_log_likelihood_info:
            neg_log_likelihood = torch.tensor(neg_log_likelihood_info["data"], dtype=DTYPE_MAP[neg_log_likelihood_info["dtype"]]).reshape(neg_log_likelihood_info["shape"])
        else:
            neg_log_likelihood = torch.rand(neg_log_likelihood_info["shape"], dtype=DTYPE_MAP[neg_log_likelihood_info["dtype"]]) * (neg_log_likelihood_info["range"][1] - neg_log_likelihood_info["range"][0]) + neg_log_likelihood_info["range"][0]
        if "data" in log_aplha_info:
            log_aplha = torch.tensor(log_aplha_info["data"], dtype=DTYPE_MAP[log_aplha_info["dtype"]]).reshape(log_aplha_info["shape"])
        else:
            log_aplha = torch.randn(log_aplha_info["shape"], dtype=DTYPE_MAP[log_aplha_info["dtype"]])
        blank = blank_info["value"]
        zero_infinity = zero_infinity_info["value"]

        input_groups.append([grad, log_probs, targets, input_lengths, target_lengths, neg_log_likelihood, log_aplha, blank, zero_infinity])
    return input_groups


def get_init_inputs():
    return []
