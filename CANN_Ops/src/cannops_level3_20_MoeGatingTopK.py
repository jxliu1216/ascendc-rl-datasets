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
import numpy as np

class Model(torch.nn.Module):

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, finished_optional: torch.Tensor, k: int):

        def softmax_func(x_np, axis=None):
            x_np = x_np.astype(np.float32)
            x_max = x_np.max(axis=axis, keepdims=True)
            x_sub = x_np - x_max
            y = np.exp(x_sub)
            x_sum = y.sum(axis=axis, keepdims=True)
            ans = y / x_sum
            return (ans, x_max, x_sum)
        gating_np = x.to(torch.float32).cpu().numpy()
        num_expert = gating_np.shape[-1]
        softmax, _, _ = softmax_func(gating_np, -1)
        indices = np.argsort(-softmax, axis=-1, kind='stable')
        indices = indices[:, :k]
        out = np.take_along_axis(softmax, indices, axis=-1)
        if finished_optional is not None:
            finished_optional_np = finished_optional.cpu().numpy()
            finished_optional_np = finished_optional_np.reshape(finished_optional_np.shape[0], 1)
            finished_optional_np = np.tile(finished_optional_np, (1, k))
            indices = np.where(finished_optional_np, num_expert, indices)
        source_row_out = np.arange(out.shape[0] * out.shape[1], dtype=np.int32).reshape([out.shape[1], out.shape[0]]).transpose(1, 0)
        return [torch.from_numpy(out).to(x.device, dtype=x.dtype), torch.from_numpy(indices).to(x.device, dtype=torch.int32), torch.from_numpy(source_row_out).to(x.device)]

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level3_20_MoeGatingTopK.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        x_info = inputs[0]
        finished_optional_info = inputs[1]
        k_info = inputs[2]

        if "data" in x_info:
            x = torch.tensor(x_info["data"], dtype=DTYPE_MAP[x_info["dtype"]]).reshape(x_info["shape"])
        else:
            x = torch.randn(x_info["shape"], dtype=DTYPE_MAP[x_info["dtype"]])
        if "data" in finished_optional_info:
            finished_optional = torch.tensor(finished_optional_info["data"], dtype=DTYPE_MAP[finished_optional_info["dtype"]]).reshape(finished_optional_info["shape"])
        else:
            finished_optional = torch.rand(finished_optional_info["shape"]) > 0.5
        k = k_info["value"]

        input_groups.append([x, finished_optional, k])
    return input_groups


def get_init_inputs():
    return []
