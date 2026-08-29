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

def _overlap_reference(bboxes_nm4: torch.Tensor, gt_nm4: torch.Tensor, eps: float, mode: str) -> torch.Tensor:
    """
    与 IouV2 kernel 一致的 CPU 参考：面积在边长上加 eps 再相乘；
    交集边长为 relu(min(x2)+eps - max(x1)) 等形式。
    bboxes_nm4: (N_a, 4), gt_nm4: (N_b, 4)，xyxy。
    返回 (N_b, N_a)，与 GE inferShape 中 overlap[gt, bbox] 一致。
    """
    a = bboxes_nm4.unsqueeze(0).float()
    b = gt_nm4.unsqueeze(1).float()
    x1a, y1a, x2a, y2a = (a[..., 0], a[..., 1], a[..., 2], a[..., 3])
    x1b, y1b, x2b, y2b = (b[..., 0], b[..., 1], b[..., 2], b[..., 3])
    wa = x2a - x1a + eps
    ha = y2a - y1a + eps
    wb = x2b - x1b + eps
    hb = y2b - y1b + eps
    area_a = wa * ha
    area_b = wb * hb
    ix1 = torch.maximum(x1a, x1b)
    iy1 = torch.maximum(y1a, y1b)
    ix2 = torch.minimum(x2a, x2b)
    iy2 = torch.minimum(y2a, y2b)
    iw = torch.relu(ix2 + eps - ix1)
    ih = torch.relu(iy2 + eps - iy1)
    inter = iw * ih
    if mode == 'iof':
        denom = area_b.clamp_min(1e-45)
    else:
        denom = (area_a + area_b - inter).clamp_min(1e-45)
    return (inter / denom).to(bboxes_nm4.dtype)

class Model(nn.Module):
    """CPU 金标准：与 op_kernel 中 IOU/IOF 公式对齐。"""

    def __init__(self):
        super().__init__()

    def forward(self, bboxes: torch.Tensor, gtboxes: torch.Tensor, mode: str, eps: float, aligned: bool) -> torch.Tensor:
        mode = mode.lower() if isinstance(mode, str) else 'iou'
        if aligned:
            boxes_a = bboxes.t().contiguous()
            boxes_b = gtboxes.t().contiguous()
            o = _overlap_reference(boxes_a, boxes_b, eps, mode)
            return torch.diagonal(o).to(bboxes.dtype).unsqueeze(1)
        return _overlap_reference(bboxes, gtboxes, eps, mode)

def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), 'cannops_level3_13_IouV2.json')
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        bboxes_info = inputs[0]
        gtboxes_info = inputs[1]
        mode_info = inputs[2]
        eps_info = inputs[3]
        aligned_info = inputs[4]

        if "data" in bboxes_info:
            bboxes = torch.tensor(bboxes_info["data"], dtype=DTYPE_MAP[bboxes_info["dtype"]]).reshape(bboxes_info["shape"])
        else:
            bboxes = torch.rand(bboxes_info["shape"], dtype=DTYPE_MAP[bboxes_info["dtype"]]) * (bboxes_info["range"][1] - bboxes_info["range"][0]) + bboxes_info["range"][0]
        if "data" in gtboxes_info:
            gtboxes = torch.tensor(gtboxes_info["data"], dtype=DTYPE_MAP[gtboxes_info["dtype"]]).reshape(gtboxes_info["shape"])
        else:
            gtboxes = torch.rand(gtboxes_info["shape"], dtype=DTYPE_MAP[gtboxes_info["dtype"]]) * (gtboxes_info["range"][1] - gtboxes_info["range"][0]) + gtboxes_info["range"][0]
        mode = mode_info["value"]
        eps = eps_info["value"]
        aligned = aligned_info["value"]

        input_groups.append([bboxes, gtboxes, mode, eps, aligned])
    return input_groups


def get_init_inputs():
    return []
