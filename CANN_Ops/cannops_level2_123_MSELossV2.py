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
import torch.nn as nn

class Model(nn.Module):
    """CPU 参考：MSE loss，与 PyTorch functional 语义一致。"""

    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, predict: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        y = (predict - label) * (predict - label)
        if self.reduction == 'sum':
            return y.sum()
        if self.reduction == 'mean':
            return y.mean()
        return y

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level2_123_MSELossV2.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        predict_info = inputs[0]
        label_info = inputs[1]

        if "data" in predict_info:
            predict = torch.tensor(predict_info["data"], dtype=DTYPE_MAP[predict_info["dtype"]]).reshape(predict_info["shape"])
        else:
            predict = torch.rand(predict_info["shape"], dtype=DTYPE_MAP[predict_info["dtype"]])
        if "data" in label_info:
            label = torch.tensor(label_info["data"], dtype=DTYPE_MAP[label_info["dtype"]]).reshape(label_info["shape"])
        else:
            label = torch.rand(label_info["shape"], dtype=DTYPE_MAP[label_info["dtype"]])

        input_groups.append([predict, label])
    return input_groups


def get_init_inputs():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level2_123_MSELossV2.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    init_groups = []
    for case in cases:
        entries = case.get("init_inputs", [])
        reduction_info = entries[0]
        reduction = reduction_info["value"]
        init_groups.append([reduction])
    return init_groups
