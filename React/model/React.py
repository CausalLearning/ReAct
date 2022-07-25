# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict

import math
import numpy as np
import pandas as pd
import torch.distributed as dist
import torch.nn as nn
from torch.nn.modules.transformer import _get_clones

from React.model.HungarianMatcher import HungarianMatcher
from React.model.roi_align import ROIAlign
from React.model.transformer import Transformer, MLP
from React.utill.misc import nested_tensor_from_tensor_list, inverse_sigmoid
from React.utill.temporal_box_producess import preprocess_groundtruth, segment_iou, ml2se, postprocessing_test_format, se2ml, segment_giou
from mmaction.models.builder import LOCALIZERS
from mmaction.models.localizers import BaseTAPGenerator

# Mostly copied from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

'''Focal loss implementation'''

import torch
import torch.nn.functional as F


def sigmoid_focal_loss(inputs, targets, num_boxes=None, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """

    bz = len(targets)
    if num_boxes is None:
        num_boxes = bz

    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.sum() / num_boxes


@LOCALIZERS.register_module()
class React(BaseTAPGenerator):
    def __init__(self,
                 input_feat_dim,  # the dimension of feature extracted by backbone
                 feat_dim=512,  # the hidden dimsnsion of the React model
                 n_head=8,  # the number of attention heads
                 num_class=20,  # the category num of the dataset
                 encoder_sample_num=3,  # the number of the sample points for the deformable attention.
                 decoder_sample_num=4,  # the number of the sample points for the deformable attention.
                 num_encoder_layers=2,  # the number of the encoder layers
                 num_decoder_layers=4,  # the number of the decoder layers
                 num_queries=40,  # the number of the query features
                 clip_len=256,  # the length (snippets number) of each clip.
                 stride_rate=0.25,  # the stride rate of the clip (for Thumos14 prediction)
                 test_bg_thershold=0.1,  # the predicted score lower than this thershold will be set as background
                 K=4,  # the number of the negative sample in contrastive learning
                 coef_l1=5.,  # the coefficient of the l1 loss
                 coef_iou=2.,  # the coefficient of the iou loss
                 coef_ce=1.,  # the coefficient of the focal loss
                 coef_aceenc=0.1,  # the coefficient of the ace-enc loss
                 coef_acedec=1.,  # the coefficient of the ace-dec loss
                 coef_quality=1.,  # the coefficient of the quality loss
                 coef_iou_decay=1.,  # the coefficient of the iou decay
                 ):
        super().__init__()

        self.feat_dim = feat_dim
        self.num_class = num_class
        self.clip_len = clip_len
        self.stride_rate = stride_rate
        self.test_bg_thershold = test_bg_thershold
        self.K = K

        self.criterion = sigmoid_focal_loss
        self.coef_l1 = coef_l1
        self.coef_iou = coef_iou
        self.coef_ce = coef_ce
        self.coef_ce_now = 0.

        self.coef_aceenc = coef_aceenc
        self.coef_acedec = coef_acedec
        self.coef_quality = coef_quality
        self.coef_iou_decay = coef_iou_decay

        # Define Module
        self.input_proj = MLP(input_feat_dim, feat_dim, feat_dim, 1)

        self.query_embed = nn.Embedding(num_queries, self.feat_dim * 2)

        self.transformer = Transformer(num_class=num_class, d_model=feat_dim, nhead=n_head, num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers, encoder_sample_num=encoder_sample_num,
                                       decoder_sample_num=decoder_sample_num, dim_feedforward=1024, normalize_before=False,
                                       return_intermediate_dec=True, dropout=0.)

        self.segment_embed = MLP(feat_dim, feat_dim, 2, 3)
        self.class_embed = nn.Linear(feat_dim, self.num_class)

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        num_pred = self.transformer.decoder.num_layers

        self.class_embed.bias.data = torch.ones(self.num_class) * bias_value
        self.class_embed = _get_clones(self.class_embed, num_pred)
        self.segment_embed = _get_clones(self.segment_embed, num_pred)
        nn.init.constant_(
            self.segment_embed[0].layers[-1].bias.data[1:], -2.0)
        # hack implementation for segment refinement
        self.transformer.decoder.segment_embed = self.segment_embed

        # predict the quality score
        self.iou_predictor = nn.Linear(feat_dim, 1)
        self.cen_predictor = nn.Linear(feat_dim, 1)

        self.HungarianMatcher = HungarianMatcher(cost_iou=5, cost_l1=2, cost_class=1)
        self.roi_extractor = ROIAlign(16, 0)

        self.cls_warmup_step = 0
        self.contrastive_count = 0.

    def forward(self,
                raw_feature,
                gt_bbox=None,
                video_gt_box=None,
                snippet_num=None,
                sample_gt=None,
                pos_feat=None,
                pos_sample_segment=None,
                neg_feat=None,
                neg_sample_segment=None,
                candidated_segments=None,
                video_meta=None,
                return_loss=True):
        """Define the computation performed at every call."""

        if return_loss:
            return self.forward_train(gt_bbox, snippet_num, raw_feature, video_gt_box, sample_gt, pos_feat, pos_sample_segment, neg_feat,
                                      neg_sample_segment, candidated_segments, video_meta)

        else:
            return self.forward_test(gt_bbox, snippet_num, raw_feature, video_gt_box, video_meta)

    @torch.no_grad()
    def forward_test(self, gt_bbox, snippet_num, raw_feature, video_gt_box, video_meta):
        # if trained with clips, stack all clip into the batch dimension, we pad the batch with the max number of clips in the batch.
        if isinstance(raw_feature[0], list):
            raw_feature = nested_tensor_from_tensor_list(raw_feature, self.clip_len, snippet_num)
        else:
            raw_feature = nested_tensor_from_tensor_list(raw_feature)

        masks = raw_feature.mask.cuda()
        input_feature = raw_feature.tensors.cuda()
        snippet_num = raw_feature.snippet_num.cuda()
        origin_snippet_num = [each["origin_snippet_num"] for each in video_meta]

        input_feature = self.input_proj(input_feature)
        raw_feature.tensors = input_feature
        raw_feature.mask = masks

        video_pred = [[[] for class_i in range(self.num_class)] for batch_i in range(len(gt_bbox))]

        # all gt for a whole video, which only used for evaluation (see the evaluate method in the dataset class)
        video_gt = preprocess_groundtruth(video_gt_box, origin_snippet_num)

        query_vector = self.query_embed.weight

        # serially predict the result for each clip and paralelly for the batch videos.
        for clip in range(input_feature.shape[1]):
            result, init_reference, inter_references, _, _ = self.transformer(input_feature[:, clip, :, :], masks[:, clip, :],
                                                                              query=query_vector,
                                                                              snippet_num=snippet_num[:, clip])  # bz, Lq, dim

            outputs_class, outputs_coord = self.output_refinement(init_reference, inter_references, result)

            pred_cls_out = outputs_class[-1]
            pred_cls_p = torch.sigmoid(pred_cls_out)
            pred_seg = outputs_coord[-1]

            # predict the quality score
            pred_iou = self.iou_predictor(result).sigmoid()[-1]
            pred_cen = self.cen_predictor(result).sigmoid()[-1]
            pred_cls_p = pred_cls_p * pred_iou * pred_cen

            clip_mask = raw_feature.clip_mask[:, clip]

            clip_pred = postprocessing_test_format(pred_seg, pred_cls_p, self.num_class, snippet_num=snippet_num[:, clip],
                                                   whole_video_snippet_num=origin_snippet_num, clip_len=self.clip_len, clip_idx=clip,
                                                   stride_rate=self.stride_rate,
                                                   threshold=self.test_bg_thershold)

            valid_clip_idx = np.where(~clip_mask)[0]

            # only the validated clips (not the padding clips) will save the results.
            for video_idx in valid_clip_idx:
                for cls in range(self.num_class):
                    if len(clip_pred[video_idx][cls]) > 0:
                        video_pred[video_idx][cls].append(clip_pred[video_idx][cls])

        # prepare the result format for the evaluator.
        null_pred = np.zeros((0, 3))
        for video_idx in range(len(video_pred)):
            for cls in range(self.num_class):
                if len(video_pred[video_idx][cls]) == 0:
                    video_pred[video_idx][cls] = null_pred
                else:
                    video_pred[video_idx][cls] = np.concatenate(video_pred[video_idx][cls], axis=0)

        return pd.DataFrame(data={'predition': video_pred, 'groundtruth': video_gt}).values

    def forward_train(self, gt_bbox, snippet_num, raw_feature, video_gt_box, sample_gt, pos_feat, pos_sample_segment, neg_feat,
                      neg_sample_segment, candidated_segments, video_meta):
        raw_feature = nested_tensor_from_tensor_list(raw_feature, self.clip_len)  # num_videos, clip_len, dim
        snippet_num = snippet_num.cuda()  # used for calculated attention offset, which is the snippet num after downsampling

        masks = raw_feature.mask.cuda()
        input_feature = raw_feature.tensors.cuda()

        device = input_feature.device
        query_vector = self.query_embed.weight

        # compute the ace-enc loss.
        if pos_feat[0].ndim > 1:  # provide contrastive data
            pos_feat = nested_tensor_from_tensor_list(pos_feat, self.clip_len)
            neg_feat = nested_tensor_from_tensor_list(neg_feat, self.clip_len)
            contrast_input_feature = torch.cat([input_feature, pos_feat.tensors.cuda(), neg_feat.tensors.cuda()],
                                               dim=0)  # 3xbatch, time, dim
            contrast_sample_gt = torch.cat(
                [sample_gt.unsqueeze(1), candidated_segments, pos_sample_segment.unsqueeze(1), neg_sample_segment.unsqueeze(1)],
                dim=1).cuda()  # bz, 1 + k(4) + 1 + 1
            h = self.input_proj(contrast_input_feature)
            contrastive_loss = self.loss_ace_enc(contrast_sample_gt, h, K=self.K)
            self.contrastive_count = self.contrastive_count + 1
            return {"loss_aceenc": contrastive_loss * self.coef_aceenc}

        input_feature = self.input_proj(input_feature)
        raw_feature.tensors = input_feature
        raw_feature.mask = masks

        result, init_reference, inter_references, memory, tgt = self.transformer(input_feature, masks, query_vector,
                                                                                 snippet_num=snippet_num)  # bz, Lq, dim

        outputs_class, outputs_coord = self.output_refinement(init_reference, inter_references, result)

        pred_iou = self.iou_predictor(result).sigmoid()
        pred_cen = self.cen_predictor(result).sigmoid()

        pred = {"pred_logits": outputs_class[-1], "pred_seg": outputs_coord[-1], 'pred_iou': pred_iou[-1],
                'pred_cen': pred_cen[-1]}

        gt_bbox = preprocess_groundtruth(gt_bbox, original_len=snippet_num, to_tensor=True, device=device)
        indices = self.HungarianMatcher(pred, gt_bbox)

        num_segs = sum(len(t["labels"]) for t in gt_bbox)
        num_segs = torch.as_tensor([num_segs], dtype=torch.float, device=device)
        num_segs = torch.clamp(num_segs, min=1).item()

        loss_dict = self.loss_segs(pred, gt_bbox, indices, num_segs)

        ce_loss = self.loss_labels(pred, gt_bbox, indices, num_segs)
        loss_dict['ce_loss'] = ce_loss * self.coef_ce_now

        # loss ace-dec
        gt_loss = self.loss_ace_dec(tgt, indices, memory, gt_bbox, memory_key_padding_mask=masks, snippet_num=snippet_num)
        loss_dict['gt_loss'] = gt_loss * self.coef_acedec

        # IoU decay
        iou_decay = self.iou_decay(pred)
        loss_dict['iou_decay'] = iou_decay * self.coef_iou_decay

        # auxiliary loss for the immediated layers
        aux_out = self._set_aux_loss(outputs_class, outputs_coord, pred_iou, pred_cen)
        aux_loss_dict = self.aux_loss(aux_out, gt_bbox, num_segs)

        loss_dict.update(aux_loss_dict)

        # trick, do not train classification head at the beginning of training
        self.cls_warmup_step = self.cls_warmup_step + 1
        if self.cls_warmup_step == 25:
            self.coef_ce_now = self.coef_ce

        return loss_dict

    def output_refinement(self, init_reference, inter_references, result):
        outputs_classes = []
        outputs_coords = []
        # gather outputs from each decoder layer
        for lvl in range(result.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]

            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](result[lvl])
            tmp = self.segment_embed[lvl](result[lvl])
            # the l-th layer (l >= 2)
            if reference.shape[-1] == 2:
                tmp += reference
            # the first layer
            else:
                assert reference.shape[-1] == 1
                tmp[..., 0] += reference[..., 0]
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)
        return outputs_class, outputs_coord

    def loss_labels(self, outputs, targets, indices, num_boxes):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)]).long()
        target_classes = torch.full(src_logits.shape[:2], self.num_class, dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        # loss_ce = self.criterion(src_logits.flatten(0, 1), target_classes.flatten(), alpha=0.75)

        targets = target_classes.flatten()
        bz = len(targets)
        device = src_logits.device
        targets = torch.zeros(bz, self.num_class + 1, device=device).scatter_(1, targets.unsqueeze(-1), 1)
        targets = targets[:, :self.num_class]
        loss_ce = self.criterion(src_logits.flatten(0, 1), targets, num_boxes)
        # loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, weight=torch.zeros(21))

        return loss_ce

    def loss_segs(self, outputs, targets, indices, num_segs):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_seg' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_segs = outputs['pred_seg'][idx]

        target_segs = torch.cat([t['segments'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_segs = F.l1_loss(src_segs, se2ml(target_segs), reduction='none')
        loss_segs = loss_segs.sum() / num_segs

        iou_cal = torch.diag(segment_giou(ml2se(src_segs), target_segs))

        loss_iou = 1 - iou_cal
        loss_iou = loss_iou.sum() / num_segs

        output_loss = {'seg_loss': loss_segs * self.coef_l1, 'iou_loss': loss_iou * self.coef_iou}

        # predict quality: regress the center of segments
        with torch.no_grad():
            src_mid = src_segs[..., 0]
            target_segs_ml = se2ml(target_segs)
            target_mid = target_segs_ml[..., 0]
            target_len = target_segs_ml[..., 1]
            broader_dis = torch.abs(src_mid - target_mid) / target_len
            broader_dis = torch.exp(-broader_dis)
            tar_cen = broader_dis.float()

        pred_cen = outputs['pred_cen'][idx].squeeze(-1)
        # loss_cen = F.l1_loss(pred_cen, tar_cen)
        loss_cen = F.binary_cross_entropy(pred_cen, tar_cen)

        # predict quality: regress the IoU value
        pred_seg_iou = segment_iou(ml2se(outputs['pred_seg'].flatten(0, 1)), target_segs).view(outputs['pred_seg'].shape[0],
                                                                                               outputs['pred_seg'].shape[1],
                                                                                               target_segs.shape[0])
        targets_start_idx = np.cumsum([len(t['segments']) for t in targets])
        targets_start_idx = np.concatenate([[0], targets_start_idx])
        tar_iou = [
            vid_seg_iou[:, targets_start_idx[i]:targets_start_idx[i + 1]].max(dim=1, keepdim=True)[0]
            if (targets_start_idx[i + 1] - targets_start_idx[i]) > 0 else torch.zeros(vid_seg_iou.shape[0], 1, device=vid_seg_iou.device)
            for i, vid_seg_iou in enumerate(pred_seg_iou)
        ]

        tar_iou = torch.stack(tar_iou, dim=0)

        pred_iou = outputs['pred_iou']
        loss_iou_regression = F.l1_loss(pred_iou, tar_iou)

        output_loss['quality_loss'] = loss_iou_regression * self.coef_quality + loss_cen * self.coef_quality

        return output_loss

    def iou_decay(self, outputs):
        src_segs = outputs['pred_seg']
        pred_segs = ml2se(src_segs)
        iou_reg_loss = segment_iou(pred_segs, pred_segs)
        iou_reg_loss = torch.triu(iou_reg_loss, diagonal=1).mean()

        return iou_reg_loss

    def loss_ace_dec(self, query_feature, indices, memory, gt_segments, memory_mask=None, memory_key_padding_mask=None, snippet_num=None):
        with torch.no_grad():
            valid_ratios = self.transformer.get_valid_ratio(memory_key_padding_mask).unsqueeze(-1)
            idx = self._get_src_permutation_idx(indices)
            target_segs = torch.cat([t['segments'][i] for t, (_, i) in zip(gt_segments, indices)], dim=0)
            target_segs = se2ml(target_segs)

            gt_num = len(target_segs)
            target_classes = torch.cat([t["labels"][J] for t, (_, J) in zip(gt_segments, indices)]).long()

            targets = torch.zeros(gt_num, self.num_class, device=target_classes.device).scatter_(1, target_classes.unsqueeze(-1), 1)
        # todo hack into decoder
        result = self.transformer.decoder.feed_gt(query_feature, memory.transpose(0, 1), memory_mask=memory_mask,
                                                  memory_key_padding_mask=memory_key_padding_mask, gt_segments=target_segs, idx=idx,
                                                  valid_ratio=valid_ratios,
                                                  snippet_num=snippet_num)

        loss_ace_dec = 0.
        # N_layer, N_gt, feat_dim
        for lvl in range(result.shape[0]):
            pred_class = self.class_embed[lvl](result[lvl])  # N_gt,
            loss_ce = self.criterion(pred_class, targets, gt_num)
            loss_ace_dec = loss_ace_dec + loss_ce

        return loss_ace_dec

    def loss_ace_enc(self, segments, memory, K=4, scale_factor=1.):
        # perform RoIAlign
        B, N = segments.shape[:2]
        memory = memory.transpose(1, 2)

        rois = self._to_roi_align_format(segments, memory.shape[2], k=K, scale_factor=scale_factor)
        roi_features = self.roi_extractor(memory, rois)
        roi_features = torch.mean(roi_features, dim=-1)
        roi_features = roi_features.view((B, N, -1))
        roi_features = F.normalize(roi_features, dim=-1)

        q = roi_features[:, :1]  # bz, 1, dim
        pos = roi_features[:, K + 1:K + 2]  # bz, 1, dim
        neg_diff_class = roi_features[:, K + 2:K + 3]  # bz, K, dim
        neg_same_class = roi_features[:, 1:K + 1]  # bz, 1, dim
        neg = torch.cat([neg_diff_class, neg_same_class], dim=1)  # bz, K+1, dim
        # neg = neg_same_class

        similarity_pos = torch.matmul(q, pos.transpose(1, 2)).squeeze(1)  # bz , 1
        similarity_neg = torch.matmul(q, neg.transpose(1, 2).detach()).squeeze(1)  # bz , K+1
        similarity = torch.cat([similarity_pos, similarity_neg], dim=-1)  # bz , 2+K

        similarity = similarity / 0.07

        labels = torch.zeros(B, dtype=torch.long, device=memory.device)  # the first vector is the positive one

        contrastive_loss = F.cross_entropy(similarity, labels)
        return contrastive_loss

    def _to_roi_align_format(self, rois, T, k=4, scale_factor=1.):
        '''Convert RoIs to RoIAlign format.
        Params:
            RoIs: normalized segments coordinates, shape (batch_size, num_segments, 4)
            T: length of the video feature sequence
        '''
        # transform to absolute axis
        B, N = rois.shape[:2]
        rois_len = rois[..., 1] - rois[..., 0]
        scale_len = (scale_factor - 1) / 2 * rois_len
        rois[..., 0] -= scale_len
        rois[..., 1] += scale_len
        rois_abs = rois * T
        # expand the RoIs
        rois_abs = torch.clamp(rois_abs, min=0, max=T)  # (N, T, 2)
        # add batch index
        batch_ind = torch.arange(0, B).view((B, 1, 1)).to(rois_abs.device)
        batch_ind = batch_ind.repeat(1, N, 1)
        batch_ind[:, k + 1] = batch_ind[:, k + 1] + B
        batch_ind[:, k + 2] = batch_ind[:, k + 2] + B * 2
        rois_abs = torch.cat((batch_ind, rois_abs.float()), dim=-1)
        # NOTE: stop gradient here to stablize training
        return rois_abs.view((B * N, 3)).detach()

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, pred_iou=None, pred_cen=None):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_seg': b, 'pred_iou': c, 'pred_cen': d} for a, b, c, d in
                zip(outputs_class[:-1], outputs_coord[:-1], pred_iou[:-1], pred_cen[:-1])]

    def aux_loss(self, aux_loss_out, gt_bbox, num_segs):
        aux_loss_dict = {}
        for i, each_pred in enumerate(aux_loss_out):
            indices = self.HungarianMatcher(each_pred, gt_bbox)

            loss_dict = self.loss_segs(each_pred, gt_bbox, indices, num_segs)
            for each_key in loss_dict.keys():
                aux_loss_dict['{}_{}'.format(i, each_key)] = loss_dict[each_key]

            ce_loss = self.loss_labels(each_pred, gt_bbox, indices, num_segs)
            aux_loss_dict['{}_ce_loss'.format(i)] = ce_loss * self.coef_ce_now

            iou_decay = self.iou_decay(each_pred)
            aux_loss_dict['{}_iou_decay'.format(i)] = iou_decay * self.coef_iou_decay

        return aux_loss_dict

    @staticmethod
    def _parse_losses(losses):
        """Parse the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary information.

        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor
                which may be a weighted sum of all losses, log_vars contains
                all the variables to be sent to the logger.
        """
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(_value for _key, _value in log_vars.items()
                   if 'loss' in _key)

        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def train_step(self, data_batch, optimizer, **kwargs):
        losses = self.forward(**data_batch)

        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(next(iter(data_batch.values()))))

        return outputs

    def val_step(self, data_batch, optimizer, **kwargs):
        # Not use method
        return dict()
