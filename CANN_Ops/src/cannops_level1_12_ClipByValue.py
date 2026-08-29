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

    def forward(self, input: torch.Tensor, clip_value_min: torch.Tensor, clip_value_max: torch.Tensor) -> torch.Tensor:
        output = torch.clamp(input, min=clip_value_min, max=clip_value_max)
        return output

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level1_12_ClipByValue.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        input_info = inputs[0]
        clip_value_min_info = inputs[1]
        clip_value_max_info = inputs[2]

        if "data" in input_info:
            input = torch.tensor(input_info["data"], dtype=DTYPE_MAP[input_info["dtype"]]).reshape(input_info["shape"])
        else:
            _dt = DTYPE_MAP[input_info["dtype"]]
            if _dt in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
                input = torch.randint(input_info["range"][0], input_info["range"][1] + 1, tuple(input_info["shape"]), dtype=_dt)
            elif _dt == torch.bool:
                input = torch.rand(input_info["shape"]) > 0.5
            else:
                input = torch.rand(input_info["shape"], dtype=_dt) * (input_info["range"][1] - input_info["range"][0]) + input_info["range"][0]
        if "data" in clip_value_min_info:
            clip_value_min = torch.tensor(clip_value_min_info["data"], dtype=DTYPE_MAP[clip_value_min_info["dtype"]]).reshape(clip_value_min_info["shape"])
        else:
            _dt = DTYPE_MAP[clip_value_min_info["dtype"]]
            if _dt in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
                clip_value_min = torch.randint(clip_value_min_info["range"][0], clip_value_min_info["range"][1] + 1, tuple(clip_value_min_info["shape"]), dtype=_dt)
            elif _dt == torch.bool:
                clip_value_min = torch.rand(clip_value_min_info["shape"]) > 0.5
            else:
                clip_value_min = torch.rand(clip_value_min_info["shape"], dtype=_dt) * (clip_value_min_info["range"][1] - clip_value_min_info["range"][0]) + clip_value_min_info["range"][0]
        if "data" in clip_value_max_info:
            clip_value_max = torch.tensor(clip_value_max_info["data"], dtype=DTYPE_MAP[clip_value_max_info["dtype"]]).reshape(clip_value_max_info["shape"])
        else:
            _dt = DTYPE_MAP[clip_value_max_info["dtype"]]
            if _dt in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
                clip_value_max = torch.randint(clip_value_max_info["range"][0], clip_value_max_info["range"][1] + 1, tuple(clip_value_max_info["shape"]), dtype=_dt)
            elif _dt == torch.bool:
                clip_value_max = torch.rand(clip_value_max_info["shape"]) > 0.5
            else:
                clip_value_max = torch.rand(clip_value_max_info["shape"], dtype=_dt) * (clip_value_max_info["range"][1] - clip_value_max_info["range"][0]) + clip_value_max_info["range"][0]

        input_groups.append([input, clip_value_min, clip_value_max])
    return input_groups


def get_init_inputs():
    return []
