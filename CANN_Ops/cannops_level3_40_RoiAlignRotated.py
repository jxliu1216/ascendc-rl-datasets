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

import math
import torch
import torch.nn as nn

def _bilinear_kernel(feat: torch.Tensor, y: float, x: float, input_h: int, input_w: int) -> torch.Tensor:
    """
    与 AscendC bilinear_interpolate 一致：越界返回 0；边界与 floor/ceil 处理与 kernel 对齐。
    feat: (H, W, C)
    """
    c = feat.shape[-1]
    device, dtype = (feat.device, feat.dtype)
    if y < -1.0 or y > float(input_h) or x < -1.0 or (x > float(input_w)):
        return torch.zeros(c, device=device, dtype=dtype)
    y = max(y, 0.0)
    x = max(x, 0.0)
    x_floor = int(math.floor(x))
    y_floor = int(math.floor(y))
    x_ceil = x_floor + 1
    y_ceil = y_floor + 1
    if x_floor >= input_w - 1:
        x_ceil = input_w - 1
        x_floor = x_ceil
        x = float(x_ceil)
    if y_floor >= input_h - 1:
        y_ceil = input_h - 1
        y_floor = y_ceil
        y = float(y_ceil)
    lx = x - float(x_floor)
    ly = y - float(y_floor)
    hx = 1.0 - lx
    hy = 1.0 - ly
    p1 = feat[y_floor, x_floor]
    p2 = feat[y_floor, x_ceil]
    p3 = feat[y_ceil, x_floor]
    p4 = feat[y_ceil, x_ceil]
    return hy * hx * p1 + hy * lx * p2 + ly * hx * p3 + ly * lx * p4

def _roi_align_rotated_reference(x: torch.Tensor, rois: torch.Tensor, pooled_h: int, pooled_w: int, spatial_scale: float, sampling_ratio: int, aligned: bool, clockwise: bool) -> torch.Tensor:
    """
    x: (B, H, W, C)，rois: (6, N) 行为 [batch_idx, cx, cy, w, h, theta]（与 kernel 平面布局一致）。
    输出: (N, pooled_h, pooled_w, C)
    """
    _, input_h, input_w, channels = x.shape
    n = rois.size(1)
    offset = -0.5 if aligned else 0.0
    out = torch.zeros((n, pooled_h, pooled_w, channels), dtype=x.dtype, device=x.device)
    xf = x.float()
    rf = rois.float()
    for j in range(n):
        b = int(rf[0, j].item())
        cx = float(rf[1, j] * spatial_scale + offset)
        cy = float(rf[2, j] * spatial_scale + offset)
        rw = float(rf[3, j] * spatial_scale)
        rh = float(rf[4, j] * spatial_scale)
        theta = rf[5, j]
        if not aligned:
            rw = max(rw, 1.0)
            rh = max(rh, 1.0)
        if clockwise:
            theta = -theta
        theta_f = float(theta)
        sin_t = math.sin(theta_f)
        cos_t = math.cos(theta_f)
        roi_start_h = -0.5 * rh
        roi_start_w = -0.5 * rw
        bin_size_h = rh / float(pooled_h)
        bin_size_w = rw / float(pooled_w)
        if sampling_ratio > 0:
            bin_grid_h = sampling_ratio
            bin_grid_w = sampling_ratio
        else:
            bin_grid_h = int(math.ceil(bin_size_h))
            bin_grid_w = int(math.ceil(bin_size_w))
            if bin_grid_h < 1:
                bin_grid_h = 1
            if bin_grid_w < 1:
                bin_grid_w = 1
        grid_h = bin_size_h / float(bin_grid_h)
        grid_w = bin_size_w / float(bin_grid_w)
        count = max(float(bin_grid_h * bin_grid_w), 1.0)
        feat_b = xf[b]
        for idx in range(pooled_h * pooled_w):
            ph = idx // pooled_w
            pw = idx - ph * pooled_w
            acc = torch.zeros(channels, dtype=torch.float32, device=x.device)
            for iy in range(bin_grid_h):
                yy = roi_start_h + ph * bin_size_h + (iy + 0.5) * grid_h
                for ix in range(bin_grid_w):
                    xx = roi_start_w + pw * bin_size_w + (ix + 0.5) * grid_w
                    y_img = yy * cos_t - xx * sin_t + float(cy)
                    x_img = yy * sin_t + xx * cos_t + float(cx)
                    acc = acc + _bilinear_kernel(feat_b, y_img, x_img, input_h, input_w).float()
            out[j, ph, pw] = (acc / count).to(dtype=x.dtype)
    return out

class Model(nn.Module):
    """CPU 金标准：与 op_kernel 中 RoiAlignRotated 采样与双线性逻辑对齐。"""

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, rois: torch.Tensor, pooled_h: int, pooled_w: int, spatial_scale: float, sampling_ratio: int, aligned: bool, clockwise: bool) -> torch.Tensor:
        return _roi_align_rotated_reference(x, rois, int(pooled_h), int(pooled_w), float(spatial_scale), int(sampling_ratio), bool(aligned), bool(clockwise))

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level3_40_RoiAlignRotated.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        x_info = inputs[0]
        rois_info = inputs[1]
        pooled_h_info = inputs[2]
        pooled_w_info = inputs[3]
        spatial_scale_info = inputs[4]
        sampling_ratio_info = inputs[5]
        aligned_info = inputs[6]
        clockwise_info = inputs[7]

        if "data" in x_info:
            x = torch.tensor(x_info["data"], dtype=DTYPE_MAP[x_info["dtype"]]).reshape(x_info["shape"])
        else:
            x = torch.randn(x_info["shape"], dtype=DTYPE_MAP[x_info["dtype"]])
        if "data" in rois_info:
            rois = torch.tensor(rois_info["data"], dtype=DTYPE_MAP[rois_info["dtype"]]).reshape(rois_info["shape"])
        else:
            rois = torch.randn(rois_info["shape"], dtype=DTYPE_MAP[rois_info["dtype"]]) * rois_info["std"] + rois_info["mean"]
        pooled_h = pooled_h_info["value"]
        pooled_w = pooled_w_info["value"]
        spatial_scale = spatial_scale_info["value"]
        sampling_ratio = sampling_ratio_info["value"]
        aligned = aligned_info["value"]
        clockwise = clockwise_info["value"]

        input_groups.append([x, rois, pooled_h, pooled_w, spatial_scale, sampling_ratio, aligned, clockwise])
    return input_groups


def get_init_inputs():
    return []
