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

from typing import List, Tuple
import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self, reduction: str):
        super(Model, self).__init__()
        self.reduction = reduction

    def forward(self, input_predict: torch.Tensor, input_label: torch.Tensor, input_dout: torch.Tensor) -> torch.Tensor:
        pred_f32 = input_predict.to(torch.float32)
        tgt_f32 = input_label.to(torch.float32)
        dout_f32 = input_dout.to(torch.float32)
        if self.reduction == 'mean':
            cof = torch.scalar_tensor(2.0 / pred_f32.numel(), device=pred_f32.device, dtype=torch.float32)
        elif self.reduction == 'sum':
            cof = torch.scalar_tensor(2.0, device=pred_f32.device, dtype=torch.float32)
        else:
            cof = torch.tensor(2.0, device=pred_f32.device, dtype=torch.float32)
        sub_res = pred_f32 - tgt_f32
        norm_grad = sub_res * cof
        golden = norm_grad * dout_f32
        return golden.to(input_predict.dtype)

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level2_129_MseLossGradV2.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        input_predict_info = inputs[0]
        input_label_info = inputs[1]
        input_dout_info = inputs[2]

        if "data" in input_predict_info:
            input_predict = torch.tensor(input_predict_info["data"], dtype=DTYPE_MAP[input_predict_info["dtype"]]).reshape(input_predict_info["shape"])
        else:
            input_predict = torch.rand(input_predict_info["shape"], dtype=DTYPE_MAP[input_predict_info["dtype"]]) * (input_predict_info["range"][1] - input_predict_info["range"][0]) + input_predict_info["range"][0]
        if "data" in input_label_info:
            input_label = torch.tensor(input_label_info["data"], dtype=DTYPE_MAP[input_label_info["dtype"]]).reshape(input_label_info["shape"])
        else:
            input_label = torch.rand(input_label_info["shape"], dtype=DTYPE_MAP[input_label_info["dtype"]]) * (input_label_info["range"][1] - input_label_info["range"][0]) + input_label_info["range"][0]
        if "data" in input_dout_info:
            input_dout = torch.tensor(input_dout_info["data"], dtype=DTYPE_MAP[input_dout_info["dtype"]]).reshape(input_dout_info["shape"])
        else:
            input_dout = torch.rand(input_dout_info["shape"], dtype=DTYPE_MAP[input_dout_info["dtype"]]) * (input_dout_info["range"][1] - input_dout_info["range"][0]) + input_dout_info["range"][0]

        input_groups.append([input_predict, input_label, input_dout])
    return input_groups


def get_init_inputs():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level2_129_MseLossGradV2.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    init_groups = []
    for case in cases:
        entries = case.get("init_inputs", [])
        reduction_info = entries[0]
        reduction = reduction_info["value"]
        init_groups.append([reduction])
    return init_groups
