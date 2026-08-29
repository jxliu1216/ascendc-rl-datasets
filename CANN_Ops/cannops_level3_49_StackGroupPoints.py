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

def _batch_idx_for_pt(pt_idx: int, ibc: torch.Tensor) -> int:
    """与 kernel 中 indices_batch_cnt 累计逻辑一致。"""
    b = ibc.numel()
    pt_cnt = int(ibc[0].item())
    bs_idx = 0
    for k in range(1, b):
        if pt_idx >= pt_cnt:
            bs_idx = k
            pt_cnt += int(ibc[k].item())
    return bs_idx

def _feature_range(bs_idx: int, fbc: torch.Tensor):
    """
    fbc 长度 B+1，与 kernel / ops-cv golden 一致：
    batch i 点数为 fbc[i]，循环用 fbc[k+1] 更新 end。
    """
    features_batch_start_idx = 0
    features_batch_end_idx = int(fbc[0].item())
    for k in range(bs_idx):
        features_batch_start_idx += int(fbc[k].item())
        features_batch_end_idx = features_batch_start_idx + int(fbc[k + 1].item())
    return (features_batch_start_idx, features_batch_end_idx)

def _stack_group_points_reference(features: torch.Tensor, features_batch_cnt: torch.Tensor, indices: torch.Tensor, indices_batch_cnt: torch.Tensor) -> torch.Tensor:
    """
    features: (N, C)；fbc / ibc: 长度 B 的 int32；
    indices: (M, nsample)；输出 (M, C, nsample)，与 infershape / kernel 索引顺序一致。
    """
    n, c = features.shape
    m, nsample = indices.shape
    b = indices_batch_cnt.numel()
    standard = m * c * nsample
    out = features.new_zeros((m, c, nsample))
    feat_flat = features.reshape(-1).float()
    fbc = features_batch_cnt.cpu()
    ibc = indices_batch_cnt.cpu()
    ind = indices.cpu().long()
    for pt_idx in range(m):
        for c_idx in range(c):
            for sample_idx in range(nsample):
                index = pt_idx * c * nsample + c_idx * nsample + sample_idx
                if index > standard:
                    continue
                bs_idx = _batch_idx_for_pt(pt_idx, ibc)
                fs, fe = _feature_range(bs_idx, fbc)
                tmp_cin = pt_idx * nsample + sample_idx
                if tmp_cin >= m * nsample:
                    continue
                cin = int(ind[pt_idx, sample_idx].item())
                in_idx = cin * c + c_idx
                if in_idx < fe * c and in_idx < n * c - fs * c:
                    fs_idx = in_idx + fs * c
                    if 0 <= fs_idx < n * c:
                        out[pt_idx, c_idx, sample_idx] = feat_flat[fs_idx]
    return out.to(dtype=features.dtype)

class Model(nn.Module):
    """CPU 金标准：与 stack_group_points kernel 中 Gather 逻辑一致。"""

    def __init__(self):
        super().__init__()

    def forward(self, features: torch.Tensor, features_batch_cnt: torch.Tensor, indices: torch.Tensor, indices_batch_cnt: torch.Tensor) -> torch.Tensor:
        return _stack_group_points_reference(features, features_batch_cnt, indices, indices_batch_cnt)

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level3_49_StackGroupPoints.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        features_info = inputs[0]
        features_batch_cnt_info = inputs[1]
        indices_info = inputs[2]
        indices_batch_cnt_info = inputs[3]

        if "data" in features_info:
            features = torch.tensor(features_info["data"], dtype=DTYPE_MAP[features_info["dtype"]]).reshape(features_info["shape"])
        else:
            features = torch.randn(features_info["shape"], dtype=DTYPE_MAP[features_info["dtype"]])
        if "data" in features_batch_cnt_info:
            features_batch_cnt = torch.tensor(features_batch_cnt_info["data"], dtype=DTYPE_MAP[features_batch_cnt_info["dtype"]]).reshape(features_batch_cnt_info["shape"])
        else:
            features_batch_cnt = torch.randint(features_batch_cnt_info["range"][0], features_batch_cnt_info["range"][1] + 1, tuple(features_batch_cnt_info["shape"]), dtype=DTYPE_MAP[features_batch_cnt_info["dtype"]])
        if "data" in indices_info:
            indices = torch.tensor(indices_info["data"], dtype=DTYPE_MAP[indices_info["dtype"]]).reshape(indices_info["shape"])
        else:
            indices = torch.randint(indices_info["range"][0], indices_info["range"][1] + 1, tuple(indices_info["shape"]), dtype=DTYPE_MAP[indices_info["dtype"]])
        if "data" in indices_batch_cnt_info:
            indices_batch_cnt = torch.tensor(indices_batch_cnt_info["data"], dtype=DTYPE_MAP[indices_batch_cnt_info["dtype"]]).reshape(indices_batch_cnt_info["shape"])
        else:
            indices_batch_cnt = torch.randint(indices_batch_cnt_info["range"][0], indices_batch_cnt_info["range"][1] + 1, tuple(indices_batch_cnt_info["shape"]), dtype=DTYPE_MAP[indices_batch_cnt_info["dtype"]])

        input_groups.append([features, features_batch_cnt, indices, indices_batch_cnt])
    return input_groups


def get_init_inputs():
    return []
