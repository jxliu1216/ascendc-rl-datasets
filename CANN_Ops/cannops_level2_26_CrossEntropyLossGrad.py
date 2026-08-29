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

from typing import List, Optional, Tuple
import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self, weight: Optional[torch.Tensor], ignore_index: int, label_smoothing: float, reduction: str, lse_square_scale_for_zloss: float):
        super(Model, self).__init__()
        self.weight = weight
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        self.lse_square_scale_for_zloss = lse_square_scale_for_zloss

    def forward(self, grad_loss: torch.Tensor, log_softmax: torch.Tensor, target: torch.Tensor, grad_zloss: Optional[torch.Tensor], lse_for_zloss: Optional[torch.Tensor]) -> List[torch.Tensor]:
        log_softmax_fp32 = log_softmax.to(torch.float32)
        grad_loss_fp32 = grad_loss.to(torch.float32)
        weight_fp32 = self.weight.to(torch.float32) if self.weight is not None else torch.ones(log_softmax.size(-1), dtype=torch.float32, device=log_softmax.device)
        target_fp32 = target.to(torch.int64)
        batch_size, num_classes = log_softmax_fp32.shape
        weight_yn = torch.gather(weight_fp32, 0, target_fp32)
        if self.ignore_index >= 0:
            ignore_mask = (target_fp32 != self.ignore_index).float()
        else:
            ignore_mask = torch.ones(batch_size, dtype=torch.float32, device=log_softmax.device)
        if self.reduction == 'mean':
            mean_out_grad = grad_loss_fp32 * (1.0 - self.label_smoothing)
            weight_sum = torch.sum(weight_yn * ignore_mask)
            loss_out_grad = mean_out_grad / (weight_sum + 1e-12)
            smooth_loss_grad = grad_loss_fp32 * self.label_smoothing / num_classes / (weight_sum + 1e-12)
            loss_out_grad = loss_out_grad.unsqueeze(-1)
            smooth_loss_grad = smooth_loss_grad.unsqueeze(-1)
        elif self.reduction == 'sum':
            sum_out_grad = grad_loss_fp32 * (1.0 - self.label_smoothing)
            loss_out_grad = sum_out_grad.unsqueeze(-1)
            smooth_loss_grad = grad_loss_fp32 * self.label_smoothing / num_classes
            smooth_loss_grad = smooth_loss_grad.unsqueeze(-1)
        else:
            none_out_grad = grad_loss_fp32 * (1.0 - self.label_smoothing)
            loss_out_grad = none_out_grad
            smooth_loss_grad = grad_loss_fp32 * self.label_smoothing / num_classes
        loss_out_grad = loss_out_grad * ignore_mask
        smooth_loss_grad = smooth_loss_grad * ignore_mask
        nll_loss_grad = loss_out_grad * weight_yn
        log_softmax_probs_grad_loss_out_sub_part = torch.exp(log_softmax_fp32) * nll_loss_grad.unsqueeze(-1)
        predictions_grad_loss_out = torch.zeros(batch_size, num_classes, dtype=torch.float32, device=log_softmax.device)
        predictions_grad_loss_out.scatter_(1, target_fp32.unsqueeze(-1), nll_loss_grad.unsqueeze(-1))
        grad_input = log_softmax_probs_grad_loss_out_sub_part - predictions_grad_loss_out
        if self.label_smoothing > 0:
            smooth_grad = smooth_loss_grad.unsqueeze(-1) * torch.ones_like(log_softmax_fp32)
            grad_input += smooth_grad
        return [grad_input.to(log_softmax.dtype)]

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level2_26_CrossEntropyLossGrad.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        grad_loss_info = inputs[0]
        log_softmax_info = inputs[1]
        target_info = inputs[2]
        grad_zloss_info = inputs[3]
        lse_for_zloss_info = inputs[4]

        if "data" in grad_loss_info:
            grad_loss = torch.tensor(grad_loss_info["data"], dtype=DTYPE_MAP[grad_loss_info["dtype"]]).reshape(grad_loss_info["shape"])
        else:
            grad_loss = torch.rand(grad_loss_info["shape"], dtype=DTYPE_MAP[grad_loss_info["dtype"]])
        if "data" in log_softmax_info:
            log_softmax = torch.tensor(log_softmax_info["data"], dtype=DTYPE_MAP[log_softmax_info["dtype"]]).reshape(log_softmax_info["shape"])
        else:
            log_softmax = torch.randn(log_softmax_info["shape"], dtype=DTYPE_MAP[log_softmax_info["dtype"]]) * log_softmax_info["std"] + log_softmax_info["mean"]
        if "data" in target_info:
            target = torch.tensor(target_info["data"], dtype=DTYPE_MAP[target_info["dtype"]]).reshape(target_info["shape"])
        else:
            target = torch.randint(target_info["range"][0], target_info["range"][1] + 1, tuple(target_info["shape"]), dtype=DTYPE_MAP[target_info["dtype"]])
        grad_zloss = None
        lse_for_zloss = None

        input_groups.append([grad_loss, log_softmax, target, grad_zloss, lse_for_zloss])
    return input_groups


def get_init_inputs():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level2_26_CrossEntropyLossGrad.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    init_groups = []
    for case in cases:
        entries = case.get("init_inputs", [])
        weight_info = entries[0]
        ignore_index_info = entries[1]
        label_smoothing_info = entries[2]
        reduction_info = entries[3]
        lse_square_scale_for_zloss_info = entries[4]
        if weight_info["type"] == "attr":
            if weight_info.get("dtype") == "none":
                weight = None
            else:
                weight = weight_info["value"]
        else:
            if "data" in weight_info:
                weight = torch.tensor(weight_info["data"], dtype=DTYPE_MAP[weight_info["dtype"]]).reshape(weight_info["shape"])
            else:
                weight = torch.rand(weight_info["shape"], dtype=DTYPE_MAP[weight_info["dtype"]])
        ignore_index = ignore_index_info["value"]
        label_smoothing = label_smoothing_info["value"]
        reduction = reduction_info["value"]
        lse_square_scale_for_zloss = lse_square_scale_for_zloss_info["value"]
        init_groups.append([weight, ignore_index, label_smoothing, reduction, lse_square_scale_for_zloss])
    return init_groups
