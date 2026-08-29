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

    def forward(self, x: torch.Tensor, window: torch.Tensor, n_fft: int, hop_length: int, win_length: int, normalized: bool, onesided: bool, return_complex: bool) -> torch.Tensor:
        if x.dtype == torch.complex64 or x.dtype == torch.complex128:
            onesided = False
        if x.dtype in (torch.float16, torch.bfloat16):
            x = x.float()
            window = window.float()
        return torch.stft(x, n_fft, hop_length=hop_length, win_length=win_length, window=window, center=False, normalized=normalized, onesided=onesided, return_complex=return_complex)

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level1_69_Stft.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        x_info = inputs[0]
        window_info = inputs[1]
        n_fft_info = inputs[2]
        hop_length_info = inputs[3]
        win_length_info = inputs[4]
        normalized_info = inputs[5]
        onesided_info = inputs[6]
        return_complex_info = inputs[7]

        if "data" in x_info:
            x = torch.tensor(x_info["data"], dtype=DTYPE_MAP[x_info["dtype"]]).reshape(x_info["shape"])
        else:
            x = torch.randn(x_info["shape"], dtype=DTYPE_MAP[x_info["dtype"]])
        if "data" in window_info:
            window = torch.tensor(window_info["data"], dtype=DTYPE_MAP[window_info["dtype"]]).reshape(window_info["shape"])
        else:
            window = torch.full(window_info["shape"], window_info["fill"], dtype=DTYPE_MAP[window_info["dtype"]])
        n_fft = n_fft_info["value"]
        hop_length = hop_length_info["value"]
        win_length = win_length_info["value"]
        normalized = normalized_info["value"]
        onesided = onesided_info["value"]
        return_complex = return_complex_info["value"]

        input_groups.append([x, window, n_fft, hop_length, win_length, normalized, onesided, return_complex])
    return input_groups


def get_init_inputs():
    return []
