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

def _reference_add_lora(y, x, weight_b, weight_a, indices, layer_idx, scale, y_offset, y_slice_size):
    out = y.clone()
    bsz = x.shape[0]
    for b in range(bsz):
        w = int(indices[b].item())
        wa = weight_a[w, layer_idx]
        wb = weight_b[w, layer_idx]
        z1 = (wa @ x[b].unsqueeze(-1)).squeeze(-1)
        z2 = (wb @ z1.unsqueeze(-1)).squeeze(-1) * scale
        sl = slice(y_offset, y_offset + y_slice_size)
        out[b, sl] = out[b, sl] + z2
    return out

class Model(nn.Module):

    def __init__(self, layer_idx: int, scale: float, y_offset: int, y_slice_size: int):
        super().__init__()
        self.layer_idx = int(layer_idx)
        self.scale = float(scale)
        self.y_offset = int(y_offset)
        self.y_slice_size = int(y_slice_size)

    def forward(self, y, x, weight_b, indices, weight_a):
        return _reference_add_lora(y, x, weight_b, weight_a, indices, self.layer_idx, self.scale, self.y_offset, self.y_slice_size)

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level1_2_AddLora.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        y_info = inputs[0]
        x_info = inputs[1]
        weight_b_info = inputs[2]
        indices_info = inputs[3]
        weight_a_info = inputs[4]

        if "data" in y_info:
            y = torch.tensor(y_info["data"], dtype=DTYPE_MAP[y_info["dtype"]]).reshape(y_info["shape"])
        else:
            y = torch.randn(y_info["shape"], dtype=DTYPE_MAP[y_info["dtype"]])
        if "data" in x_info:
            x = torch.tensor(x_info["data"], dtype=DTYPE_MAP[x_info["dtype"]]).reshape(x_info["shape"])
        else:
            x = torch.randn(x_info["shape"], dtype=DTYPE_MAP[x_info["dtype"]])
        if "data" in weight_b_info:
            weight_b = torch.tensor(weight_b_info["data"], dtype=DTYPE_MAP[weight_b_info["dtype"]]).reshape(weight_b_info["shape"])
        else:
            weight_b = torch.randn(weight_b_info["shape"], dtype=DTYPE_MAP[weight_b_info["dtype"]])
        if "data" in indices_info:
            indices = torch.tensor(indices_info["data"], dtype=DTYPE_MAP[indices_info["dtype"]]).reshape(indices_info["shape"])
        else:
            indices = torch.randint(indices_info["range"][0], indices_info["range"][1] + 1, tuple(indices_info["shape"]), dtype=DTYPE_MAP[indices_info["dtype"]])
        if "data" in weight_a_info:
            weight_a = torch.tensor(weight_a_info["data"], dtype=DTYPE_MAP[weight_a_info["dtype"]]).reshape(weight_a_info["shape"])
        else:
            weight_a = torch.randn(weight_a_info["shape"], dtype=DTYPE_MAP[weight_a_info["dtype"]])

        input_groups.append([y, x, weight_b, indices, weight_a])
    return input_groups


def get_init_inputs():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level1_2_AddLora.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    init_groups = []
    for case in cases:
        entries = case.get("init_inputs", [])
        layer_idx_info = entries[0]
        scale_info = entries[1]
        y_offset_info = entries[2]
        y_slice_size_info = entries[3]
        layer_idx = layer_idx_info["value"]
        scale = scale_info["value"]
        y_offset = y_offset_info["value"]
        y_slice_size = y_slice_size_info["value"]
        init_groups.append([layer_idx, scale, y_offset, y_slice_size])
    return init_groups
