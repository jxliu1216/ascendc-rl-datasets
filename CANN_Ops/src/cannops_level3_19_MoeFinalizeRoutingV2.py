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

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, expandedX: torch.Tensor, expandedRowIdx: torch.Tensor, x1Optional: torch.Tensor, x2Optional: torch.Tensor, biasOptional: torch.Tensor, scalesOptional: torch.Tensor, expertIdxOptional: torch.Tensor, dropPadMode: int) -> torch.Tensor:
        if len(expandedX.shape) == 2:
            num_rows = expertIdxOptional.shape[0]
            k = expertIdxOptional.shape[1]
            hidden_size = expandedX.shape[-1]
            output = torch.zeros(num_rows, hidden_size, device=expandedX.device, dtype=expandedX.dtype)
            has_bias = biasOptional.numel() > 0
            for i in range(num_rows):
                temp_sum = torch.zeros(hidden_size, device=expandedX.device, dtype=expandedX.dtype)
                for j in range(k):
                    temp_sum += scalesOptional[i, j] * expandedX[expandedRowIdx[i + j * num_rows]]
                    if has_bias:
                        expert_id = expertIdxOptional[i, j].item()
                        temp_sum += scalesOptional[i, j] * biasOptional[expert_id]
                output[i] = x1Optional[i] + x2Optional[i] + temp_sum
            return output

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level3_19_MoeFinalizeRoutingV2.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        expandedX_info = inputs[0]
        expandedRowIdx_info = inputs[1]
        x1Optional_info = inputs[2]
        x2Optional_info = inputs[3]
        biasOptional_info = inputs[4]
        scalesOptional_info = inputs[5]
        expertIdxOptional_info = inputs[6]
        dropPadMode_info = inputs[7]

        if "data" in expandedX_info:
            expandedX = torch.tensor(expandedX_info["data"], dtype=DTYPE_MAP[expandedX_info["dtype"]]).reshape(expandedX_info["shape"])
        else:
            expandedX = torch.randn(expandedX_info["shape"], dtype=DTYPE_MAP[expandedX_info["dtype"]])
        if "data" in expandedRowIdx_info:
            expandedRowIdx = torch.tensor(expandedRowIdx_info["data"], dtype=DTYPE_MAP[expandedRowIdx_info["dtype"]]).reshape(expandedRowIdx_info["shape"])
        else:
            expandedRowIdx = torch.randint(expandedRowIdx_info["range"][0], expandedRowIdx_info["range"][1] + 1, tuple(expandedRowIdx_info["shape"]), dtype=DTYPE_MAP[expandedRowIdx_info["dtype"]])
        if "data" in x1Optional_info:
            x1Optional = torch.tensor(x1Optional_info["data"], dtype=DTYPE_MAP[x1Optional_info["dtype"]]).reshape(x1Optional_info["shape"])
        else:
            x1Optional = torch.randn(x1Optional_info["shape"], dtype=DTYPE_MAP[x1Optional_info["dtype"]])
        if "data" in x2Optional_info:
            x2Optional = torch.tensor(x2Optional_info["data"], dtype=DTYPE_MAP[x2Optional_info["dtype"]]).reshape(x2Optional_info["shape"])
        else:
            x2Optional = torch.randn(x2Optional_info["shape"], dtype=DTYPE_MAP[x2Optional_info["dtype"]])
        if "data" in biasOptional_info:
            biasOptional = torch.tensor(biasOptional_info["data"], dtype=DTYPE_MAP[biasOptional_info["dtype"]]).reshape(biasOptional_info["shape"])
        else:
            biasOptional = torch.randn(biasOptional_info["shape"], dtype=DTYPE_MAP[biasOptional_info["dtype"]])
        if "data" in scalesOptional_info:
            scalesOptional = torch.tensor(scalesOptional_info["data"], dtype=DTYPE_MAP[scalesOptional_info["dtype"]]).reshape(scalesOptional_info["shape"])
        else:
            scalesOptional = torch.randn(scalesOptional_info["shape"], dtype=DTYPE_MAP[scalesOptional_info["dtype"]])
        if "data" in expertIdxOptional_info:
            expertIdxOptional = torch.tensor(expertIdxOptional_info["data"], dtype=DTYPE_MAP[expertIdxOptional_info["dtype"]]).reshape(expertIdxOptional_info["shape"])
        else:
            expertIdxOptional = torch.randint(expertIdxOptional_info["range"][0], expertIdxOptional_info["range"][1] + 1, tuple(expertIdxOptional_info["shape"]), dtype=DTYPE_MAP[expertIdxOptional_info["dtype"]])
        dropPadMode = dropPadMode_info["value"]

        input_groups.append([expandedX, expandedRowIdx, x1Optional, x2Optional, biasOptional, scalesOptional, expertIdxOptional, dropPadMode])
    return input_groups


def get_init_inputs():
    return []
