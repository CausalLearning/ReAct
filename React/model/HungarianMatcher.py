# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from React.utill.temporal_box_producess import segment_iou, ml2se


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_l1: float = 1, cost_iou: float = 1):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_l1 = cost_l1
        self.cost_iou = cost_iou
        assert cost_class != 0 or cost_l1 != 0 or cost_iou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching
        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 2] with the predicted box coordinates
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "segmentation": Tensor of dim [num_target_boxes, 2] containing the target box coordinates
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]

        out_seg = outputs["pred_seg"]  # [batch_size * num_queries, 2]
        out_seg = ml2se(out_seg.flatten(0, 1)).view(out_seg.shape)

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets], dim=0).long()
        tgt_segs = torch.cat([v["segments"] for v in targets], dim=0).float()

        out_seg = out_seg.flatten(0, 1)

        # tgt_bbox = torch.tensor(tgt_bbox, dtype=torch.float, device=out_seg.device)

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        cost_class = -out_prob[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_l1 = torch.cdist(out_seg, tgt_segs, p=1)

        # Compute the iou cost betwen boxes
        cost_iou = -segment_iou(out_seg, tgt_segs)

        # Final cost matrix
        C = self.cost_l1 * cost_l1 + self.cost_class * cost_class + self.cost_iou * cost_iou
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["segments"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64, device=out_prob.device), torch.as_tensor(j, dtype=torch.int64, device=out_prob.device)) for i, j in indices]
