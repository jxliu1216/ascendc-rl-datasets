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
import torch.nn.functional as F

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, scale_value: float, head_num: int) -> torch.Tensor:
        query = query.to(torch.float32)
        key = key.to(torch.float32)
        value = value.to(torch.float32)
        scores = torch.matmul(query, key.transpose(-1, -2))
        scores = scores * scale_value
        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, value)
        return output.to(torch.float16)

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level3_4_FlashAttentionScoreWithLargeHeadDim.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        query_info = inputs[0]
        key_info = inputs[1]
        value_info = inputs[2]
        scale_value_info = inputs[3]
        head_num_info = inputs[4]

        if "data" in query_info:
            query = torch.tensor(query_info["data"], dtype=DTYPE_MAP[query_info["dtype"]]).reshape(query_info["shape"])
        else:
            query = torch.rand(query_info["shape"], dtype=DTYPE_MAP[query_info["dtype"]]) * (query_info["range"][1] - query_info["range"][0]) + query_info["range"][0]
        if "data" in key_info:
            key = torch.tensor(key_info["data"], dtype=DTYPE_MAP[key_info["dtype"]]).reshape(key_info["shape"])
        else:
            key = torch.rand(key_info["shape"], dtype=DTYPE_MAP[key_info["dtype"]]) * (key_info["range"][1] - key_info["range"][0]) + key_info["range"][0]
        if "data" in value_info:
            value = torch.tensor(value_info["data"], dtype=DTYPE_MAP[value_info["dtype"]]).reshape(value_info["shape"])
        else:
            value = torch.rand(value_info["shape"], dtype=DTYPE_MAP[value_info["dtype"]]) * (value_info["range"][1] - value_info["range"][0]) + value_info["range"][0]
        scale_value = scale_value_info["value"]
        head_num = head_num_info["value"]

        input_groups.append([query, key, value, scale_value, head_num])
    return input_groups


def get_init_inputs():
    return []
