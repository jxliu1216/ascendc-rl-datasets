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
import math

class Model(nn.Module):
    """
    实现FeedsRepeat算子功能的模型。
    """

    def __init__(self):
        """
        初始化模型。
        """
        super(Model, self).__init__()

    def forward(self, feeds: torch.Tensor, feeds_repeat_times: torch.Tensor, output_feeds_size: int) -> torch.Tensor:
        """
        实现FeedsRepeat算子功能。

        Args:
            feeds: 输入张量
            feeds_repeat_times: 重复次数张量
            output_feeds_size: 输出的feeds大小

        Returns:
            处理后的输出张量
        """
        repeated = torch.repeat_interleave(feeds, feeds_repeat_times, dim=0)
        total_repeated = feeds_repeat_times.sum().item()
        pad_size = output_feeds_size - total_repeated
        if pad_size > 0:
            output_shape = (output_feeds_size,) + feeds.shape[1:]
            output = torch.zeros(output_shape, dtype=feeds.dtype, device=feeds.device)
            output[:total_repeated] = repeated
            return output
        else:
            return repeated

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level2_41_FeedsRepeat.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        feeds_info = inputs[0]
        feeds_repeat_times_info = inputs[1]
        output_feeds_size_info = inputs[2]

        if "data" in feeds_info:
            feeds = torch.tensor(feeds_info["data"], dtype=DTYPE_MAP[feeds_info["dtype"]]).reshape(feeds_info["shape"])
        else:
            feeds = torch.randn(feeds_info["shape"], dtype=DTYPE_MAP[feeds_info["dtype"]])
        if "data" in feeds_repeat_times_info:
            feeds_repeat_times = torch.tensor(feeds_repeat_times_info["data"], dtype=DTYPE_MAP[feeds_repeat_times_info["dtype"]]).reshape(feeds_repeat_times_info["shape"])
        else:
            feeds_repeat_times = torch.randint(feeds_repeat_times_info["range"][0], feeds_repeat_times_info["range"][1] + 1, tuple(feeds_repeat_times_info["shape"]), dtype=DTYPE_MAP[feeds_repeat_times_info["dtype"]])
        output_feeds_size = output_feeds_size_info["value"]

        input_groups.append([feeds, feeds_repeat_times, output_feeds_size])
    return input_groups


def get_init_inputs():
    return []
