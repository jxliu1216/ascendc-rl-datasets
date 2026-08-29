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

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, sel: torch.Tensor, input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
        input1_sel = input1 * sel
        input2_sel = input2 * ~sel
        reduce_res = input1_sel + input2_sel
        max_res = torch.amax(reduce_res, dim=-1, keepdim=True)
        sub_res = reduce_res - max_res
        exp_res = torch.exp(sub_res)
        sum_res = torch.sum(exp_res, dim=-1, keepdim=True)
        output = exp_res / sum_res
        return output

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level2_141_SelectReduceMaxDSubExpReduceSumDRealDiv.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        sel_info = inputs[0]
        input1_info = inputs[1]
        input2_info = inputs[2]

        if "data" in sel_info:
            sel = torch.tensor(sel_info["data"], dtype=DTYPE_MAP[sel_info["dtype"]]).reshape(sel_info["shape"])
        else:
            sel = torch.rand(sel_info["shape"]) > 0.5
        if "data" in input1_info:
            input1 = torch.tensor(input1_info["data"], dtype=DTYPE_MAP[input1_info["dtype"]]).reshape(input1_info["shape"])
        else:
            input1 = torch.randn(input1_info["shape"], dtype=DTYPE_MAP[input1_info["dtype"]])
        if "data" in input2_info:
            input2 = torch.tensor(input2_info["data"], dtype=DTYPE_MAP[input2_info["dtype"]]).reshape(input2_info["shape"])
        else:
            input2 = torch.randn(input2_info["shape"], dtype=DTYPE_MAP[input2_info["dtype"]])

        input_groups.append([sel, input1, input2])
    return input_groups


def get_init_inputs():
    return []
