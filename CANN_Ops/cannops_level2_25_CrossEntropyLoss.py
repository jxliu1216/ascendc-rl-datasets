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

    def __init__(self, weight: Optional[torch.Tensor], ignore_index: int, label_smoothing: float, reduction: str, lse_square_scale_for_zloss: float, return_zloss: bool):
        super(Model, self).__init__()
        self.weight = weight
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        self.lse_square_scale_for_zloss = lse_square_scale_for_zloss
        self.return_zloss = return_zloss

    def forward(self, input_predictions: torch.Tensor, target_labels: torch.Tensor) -> List[torch.Tensor]:
        n, c = input_predictions.shape
        input_dtype = input_predictions.dtype
        predictions_fp32 = input_predictions.to(torch.float32)
        if self.weight is None:
            weight_fp32 = torch.ones((c,), dtype=torch.float32, device=predictions_fp32.device)
        else:
            weight_fp32 = self.weight.to(torch.float32)
        predictions_max = torch.max(predictions_fp32, dim=1, keepdim=True)[0]
        lse = predictions_max + torch.log(torch.sum(torch.exp(predictions_fp32 - predictions_max), dim=1, keepdim=True))
        log_softmax_probs = predictions_fp32 - lse
        nll_loss_terms = torch.gather(log_softmax_probs, 1, target_labels.unsqueeze(-1)).squeeze(-1)
        weight_for_targets = torch.gather(weight_fp32, 0, target_labels)
        loss_out_unreduced = -nll_loss_terms * weight_for_targets
        if self.ignore_index >= 0:
            ignore_mask = (target_labels != self.ignore_index).float()
            loss_out_unreduced = loss_out_unreduced * ignore_mask
        else:
            ignore_mask = torch.ones((n,), dtype=torch.float32, device=predictions_fp32.device)
        smooth_loss_unreduced = -torch.sum(log_softmax_probs * weight_fp32.unsqueeze(0), dim=1, keepdim=False)
        if self.ignore_index >= 0:
            smooth_loss_unreduced = smooth_loss_unreduced * ignore_mask
        weight_after_mask_sum = torch.sum(weight_for_targets * ignore_mask, dim=-1, keepdim=False)
        base_loss_reduced = None
        if self.reduction == 'mean':
            base_loss_reduced = torch.sum(loss_out_unreduced, dim=-1, keepdim=False) / (weight_after_mask_sum + 1e-12)
        elif self.reduction == 'sum':
            base_loss_reduced = torch.sum(loss_out_unreduced, dim=-1, keepdim=False)
        else:
            base_loss_reduced = loss_out_unreduced
        smoothed_term_reduced = None
        if self.reduction == 'mean':
            smoothed_term_reduced = torch.sum(smooth_loss_unreduced, dim=-1, keepdim=False) / (weight_after_mask_sum + 1e-12) * self.label_smoothing / c
        elif self.reduction == 'sum':
            smoothed_term_reduced = torch.sum(smooth_loss_unreduced, dim=-1, keepdim=False) * self.label_smoothing / c
        else:
            smoothed_term_reduced = smooth_loss_unreduced * self.label_smoothing / c
        loss_out = (1 - self.label_smoothing) * base_loss_reduced + smoothed_term_reduced
        zloss_out_dtype = input_dtype if input_dtype in [torch.float16, torch.bfloat16] else torch.float32
        zloss_out = torch.zeros((1,), dtype=zloss_out_dtype, device=predictions_fp32.device)
        lse_for_zloss_out = lse.squeeze(-1)
        if self.return_zloss:
            zloss_out = self.lse_square_scale_for_zloss * torch.mean(lse.pow(2))
            zloss_out = zloss_out.reshape(1)
        return [loss_out.to(input_dtype), log_softmax_probs.to(input_dtype), zloss_out.to(input_dtype), lse_for_zloss_out.to(input_dtype)]

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level2_25_CrossEntropyLoss.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        input_predictions_info = inputs[0]
        target_labels_info = inputs[1]

        if "data" in input_predictions_info:
            input_predictions = torch.tensor(input_predictions_info["data"], dtype=DTYPE_MAP[input_predictions_info["dtype"]]).reshape(input_predictions_info["shape"])
        else:
            input_predictions = torch.randn(input_predictions_info["shape"], dtype=DTYPE_MAP[input_predictions_info["dtype"]])
        if "data" in target_labels_info:
            target_labels = torch.tensor(target_labels_info["data"], dtype=DTYPE_MAP[target_labels_info["dtype"]]).reshape(target_labels_info["shape"])
        else:
            target_labels = torch.randint(target_labels_info["range"][0], target_labels_info["range"][1] + 1, tuple(target_labels_info["shape"]), dtype=DTYPE_MAP[target_labels_info["dtype"]])

        input_groups.append([input_predictions, target_labels])
    return input_groups


def get_init_inputs():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level2_25_CrossEntropyLoss.json')
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
        return_zloss_info = entries[5]
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
        return_zloss = return_zloss_info["value"]
        init_groups.append([weight, ignore_index, label_smoothing, reduction, lse_square_scale_for_zloss, return_zloss])
    return init_groups
