"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
Modules to compute the matching cost and solve the corresponding LSAP.

Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F 

from scipy.optimize import linear_sum_assignment
from typing import Dict 

from .box_ops import box_cxcywh_to_xyxy, generalized_box_iou, nwd_matrix

from ...core import register


@register()
class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    __share__ = ['use_focal_loss', ]

    def __init__(self, weight_dict, use_focal_loss=False, alpha=0.25, gamma=2.0):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = weight_dict['cost_class']
        self.cost_bbox = weight_dict['cost_bbox']
        self.cost_giou = weight_dict['cost_giou']
        self.cost_nwd = weight_dict.get('cost_nwd', 2.0)

        self.use_focal_loss = use_focal_loss
        self.alpha = alpha
        self.gamma = gamma

        assert self.cost_class != 0 or self.cost_bbox != 0 or self.cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs: Dict[str, torch.Tensor], targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        if self.use_focal_loss:
            out_prob = F.sigmoid(outputs["pred_logits"].flatten(0, 1))
        else:
            out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]

        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        if self.use_focal_loss:
            out_prob = out_prob[:, tgt_ids]
            neg_cost_class = (1 - self.alpha) * (out_prob ** self.gamma) * (-(1 - out_prob + 1e-8).log())
            pos_cost_class = self.alpha * ((1 - out_prob) ** self.gamma) * (-(out_prob + 1e-8).log())
            cost_class = pos_cost_class - neg_cost_class        
        else:
            cost_class = -out_prob[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # ====== 【SADM 架构核心：尺度自适应匹配】 ======
        # 1. 计算 GT 归一化物理面积
        tgt_area = tgt_bbox[:, 2] * tgt_bbox[:, 3]
        # 2. 生成指数衰减权重 (0.002 是一个黄金阈值：面积越小，权重越接近1)
        # 对于 30x30 的目标，权重极高；对于 96x96 以上的大目标，权重指数级衰减至近乎 0
        tiny_weight = torch.exp(-tgt_area / 0.002).unsqueeze(0)  # shape: [1, num_tgt]

        # 3. 分别计算两种代价
        cost_nwd = 1.0 - nwd_matrix(out_bbox, tgt_bbox, C=0.015)
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        # 4. 动态软融合（绝不一刀切 0.5+0.5，让网络自己看菜下饭！）
        combined_box_cost = tiny_weight * cost_nwd + (1.0 - tiny_weight) * cost_giou

        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * combined_box_cost
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        indices = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

        return {'indices': indices}

        # # Compute the giou cost betwen boxes
        # # cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))
        #
        # # 注意: out_bbox 和 tgt_bbox 本身就是 cxcywh 格式，直接传给 nwd_matrix
        # cost_nwd = 1.0 - nwd_matrix(out_bbox, tgt_bbox, C=0.015)
        #
        # # Final cost matrix: 匹配代价值仅由 分类代价 + L1绝对距离代价 + NWD分布代价 组成。
        # # 这样即使小目标预测框偏离很远（GIoU为0），NWD依然能提供极其平滑的代价值引导分配！
        # C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_nwd * cost_nwd
        # C = C.view(bs, num_queries, -1).cpu()
        # # 【修改】Final cost matrix: 将 GIoU 和 NWD 结合起来进行匹配
        # # 这里权重 0.5 可以根据需要调整，意味着匹配时同时看重边缘重合度(GIoU)和中心距离(NWD)
        # # combined_box_cost = 0.5 * cost_giou + 0.5 * cost_nwd
        #
        # # Final cost matrix
        # # C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * combined_box_cost
        # # C = C.view(bs, num_queries, -1).cpu()
        #
        # sizes = [len(v["boxes"]) for v in targets]
        # indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        # indices = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
        #
        # return {'indices': indices}
        