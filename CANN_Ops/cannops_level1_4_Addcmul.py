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

    def forward(self, input_data: torch.Tensor, input_x1: torch.Tensor, input_x2: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        if input_data.dtype == torch.bfloat16:
            output = input_data.to(torch.float32) + input_x1.to(torch.float32) * input_x2.to(torch.float32) * value.to(torch.float32)
            output = output.to(input_data.dtype)
        else:
            output = input_data + input_x1 * input_x2 * value
        return output

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level1_4_Addcmul.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        input_data_info = inputs[0]
        input_x1_info = inputs[1]
        input_x2_info = inputs[2]
        value_info = inputs[3]

        if "data" in input_data_info:
            input_data = torch.tensor(input_data_info["data"], dtype=DTYPE_MAP[input_data_info["dtype"]]).reshape(input_data_info["shape"])
        else:
            input_data = torch.rand(input_data_info["shape"], dtype=DTYPE_MAP[input_data_info["dtype"]]) * (input_data_info["range"][1] - input_data_info["range"][0]) + input_data_info["range"][0]
        if "data" in input_x1_info:
            input_x1 = torch.tensor(input_x1_info["data"], dtype=DTYPE_MAP[input_x1_info["dtype"]]).reshape(input_x1_info["shape"])
        else:
            input_x1 = torch.rand(input_x1_info["shape"], dtype=DTYPE_MAP[input_x1_info["dtype"]]) * (input_x1_info["range"][1] - input_x1_info["range"][0]) + input_x1_info["range"][0]
        if "data" in input_x2_info:
            input_x2 = torch.tensor(input_x2_info["data"], dtype=DTYPE_MAP[input_x2_info["dtype"]]).reshape(input_x2_info["shape"])
        else:
            input_x2 = torch.rand(input_x2_info["shape"], dtype=DTYPE_MAP[input_x2_info["dtype"]]) * (input_x2_info["range"][1] - input_x2_info["range"][0]) + input_x2_info["range"][0]
        if "data" in value_info:
            value = torch.tensor(value_info["data"], dtype=DTYPE_MAP[value_info["dtype"]]).reshape(value_info["shape"])
        else:
            value = torch.full(value_info["shape"], value_info["fill"], dtype=DTYPE_MAP[value_info["dtype"]])

        input_groups.append([input_data, input_x1, input_x2, value])
    return input_groups


def get_init_inputs():
    return []
