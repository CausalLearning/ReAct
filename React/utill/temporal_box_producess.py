import numpy as np
import pandas as pd
import torch


def mid_dis2start_end(x):
    # x shape (num_seg, 2)
    if isinstance(x, torch.Tensor):
        mid, dis = x[:, 0].unsqueeze(-1), x[:, 1].unsqueeze(-1)
        b = [(mid - 0.5 * dis).int(), (mid + 0.5 * dis).int()]
        return torch.cat(b, dim=-1)
    else:
        mid, dis = x[:, 0, None], x[:, 1, None]
        b = [(mid - 0.5 * dis), (mid + 0.5 * dis)]
        return np.concatenate(b, axis=-1)


def start_end2mid_dis(x):
    # x shape (num_seg, 2)
    s, e = x[:, 0], x[:, 1]
    b = [((s + e) / 2).astype(np.int), (e - s).astype(np.int)]
    return np.stack(b)


def find_match_gt(seg_instance, gt):
    # seg_instance: [N_action, 2(start_time/frame, end_time)]
    # gt : [N_gt, 3(start_time/frame, end_time,class)]
    seg_instance = seg_instance.unsqueeze(1)
    gt = gt.unsqueeze(0)
    inter = torch.max(torch.zeros(1, device=gt.device),
                      torch.min(gt[:, :, 1], seg_instance[:, :, 1]) - torch.max(gt[:, :, 0], seg_instance[:, :, 0]))
    union = (gt[:, :, 1] - gt[:, :, 0]) + (seg_instance[:, :, 1] - seg_instance[:, :, 0]) - inter
    iou = inter / union

    idx = torch.argmax(iou, dim=-1)
    gt_class = gt[0, idx, 2]
    return gt_class


def stack_predicted_data(video_pred):
    stack_data = []
    for class_id in range(len(video_pred)):
        if len(video_pred[class_id]) > 0:
            data_i = np.concatenate([video_pred[class_id], np.ones((len(video_pred[class_id]), 1)) * class_id], axis=-1)
            stack_data.append(data_i)

    if len(stack_data) == 0:
        return np.zeros((0, 3))

    stack_data = np.concatenate(stack_data, axis=0)
    return stack_data


def nms(vid_dets, thresh):
    for vid, pred_out in enumerate(vid_dets):
        out = [np.zeros((0, 3)) for _ in range(len(vid_dets[0]))]

        dets = stack_predicted_data(pred_out)

        if len(dets) == 0:
            vid_dets[vid] = out
            continue

        start = dets[:, 0]
        end = dets[:, 1]
        scores = dets[:, 2]
        areas = (end - start)
        order = scores.argsort()[::-1]
        while order.size > 0:
            i = order[0]
            keep = dets[i]
            out[int(keep[3])] = np.concatenate([out[int(keep[3])], keep[None, :3]], axis=0)
            xx1 = np.maximum(start[i], start[order[1:]])
            xx2 = np.minimum(end[i], end[order[1:]])
            inter = np.maximum(0.0, xx2 - xx1)
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]

        vid_dets[vid] = out


def softnms_v2(vid_dets, sigma=0.5, top_k=1000, score_threshold=0.):
    for vid, pred_out in enumerate(vid_dets):
        out = [np.zeros((0, 3)) for _ in range(len(vid_dets[0]))]

        segments = stack_predicted_data(pred_out)
        if len(segments) == 0:
            continue

        tstart = segments[:, 0]
        tend = segments[:, 1]
        tscore = segments[:, 2]
        tcls = segments[:, 3]
        done_mask = tscore < -1  # set all to False
        undone_mask = tscore >= score_threshold
        while undone_mask.sum() > 0 and done_mask.sum() < top_k:
            idx = tscore[undone_mask].argmax()
            idx = undone_mask.nonzero()[0][idx].item()

            undone_mask[idx] = False
            done_mask[idx] = True

            top_start = tstart[idx]
            top_end = tend[idx]
            _tstart = tstart[undone_mask]
            _tend = tend[undone_mask]
            tt1 = _tstart.clip(min=top_start)
            tt2 = _tend.clip(max=top_end)
            intersection = (tt2 - tt1).clip(min=0)
            duration = _tend - _tstart
            tmp_width = max(top_end - top_start, 1e-5)
            iou = intersection / (tmp_width + duration - intersection)
            scales = np.exp(-iou ** 2 / sigma)
            tscore[undone_mask] *= scales
            undone_mask[tscore < score_threshold] = False

        out_start = tstart[done_mask]
        out_end = tend[done_mask]
        out_score = tscore[done_mask]
        out_cls = tcls[done_mask]
        for i in range(len(out_cls)):
            out[int(out_cls[i])] = np.concatenate([out[int(out_cls[i])], np.array([[out_start[i], out_end[i], out_score[i]]])], axis=0)

        vid_dets[vid] = out


def postprocessing_test_format(output, pred_prob, cls_num, snippet_num, whole_video_snippet_num, clip_len, clip_idx, stride_rate,
                               threshold=0.1):
    # predicte results for all clip and re-normalized all the coordinates into the whole video coordinates
    # we predicted all the clip results parallelly for a batch of videos. We pad the number of clip for each video to the max number in the
    # batch of video, and parallelly predict the clip at the same location for the batch.

    # pred_prob: batch_size, max_action_num
    batch_pred = []
    if isinstance(whole_video_snippet_num, list):
        whole_video_snippet_num = np.array(whole_video_snippet_num)
    elif isinstance(whole_video_snippet_num, torch.Tensor):
        whole_video_snippet_num = whole_video_snippet_num.cpu().numpy()

    # the ratio of the clip length and the whole video length
    window_ratio = clip_len / whole_video_snippet_num
    # the start location (in each videos) for each clip.
    start_ratio = clip_idx * window_ratio * stride_rate
    # the validated ratio in the clip for each video.
    validated_clip_ratio = snippet_num.cpu().numpy() / whole_video_snippet_num

    prob, pred_idx = pred_prob.topk(1, dim=-1)
    output = output.cpu().numpy()
    prob = prob.cpu().numpy()
    pred_idx = pred_idx.cpu().numpy()

    for v_out, v_pred_idx, v_pred_prob, v_clip_ratio, v_start_ratio in zip(output, pred_idx, prob, validated_clip_ratio, start_ratio):
        det_result = []

        # the predicted results in the range of [0,1]
        v_out_start = np.clip(v_out[..., 0] - v_out[..., 1] / 2, a_min=0, a_max=1)
        v_out_end = np.clip(v_out[..., 0] + v_out[..., 1] / 2, a_min=0, a_max=1)

        # re-normalized to global coordinates
        v_out_start = v_out_start * v_clip_ratio + v_start_ratio
        v_out_end = v_out_end * v_clip_ratio + v_start_ratio

        v_pred_idx = v_pred_idx.squeeze(-1)
        v_pred_prob = v_pred_prob.squeeze(-1)

        # select predicted segment for each class, prepare the data format for the evaluator.
        for each_cls in range(cls_num):
            cls_out = np.zeros((0, 3))
            cls_pred_idx = v_pred_idx == each_cls  # Nq, 1
            seg_start = v_out_start[cls_pred_idx, None]
            seg_end = v_out_end[cls_pred_idx, None]

            if len(v_pred_prob[cls_pred_idx]) == 0:
                det_result.append(cls_out)
                continue

            # keep the segments with the score over the threshold
            not_bg_idx = v_pred_prob[cls_pred_idx] > threshold
            seg_start = seg_start[not_bg_idx]
            seg_end = seg_end[not_bg_idx]
            prob_left = v_pred_prob[cls_pred_idx][not_bg_idx, None]

            cls_out = np.concatenate([seg_start, seg_end, prob_left], axis=-1)  # (N_action, start, end, probability)
            det_result.append(cls_out)

        batch_pred.append(det_result)

    return batch_pred


def segment_iou(segment1, segment2):
    assert (segment1[..., 1] >= segment1[..., 0]).all()
    assert (segment2[..., 1] >= segment2[..., 0]).all()

    if segment1.ndim == 2 and segment2.ndim == 2:
        inter = torch.max(torch.zeros(1, device=segment1.device),
                          torch.min(segment1[:, None, 1], segment2[None, :, 1]) - torch.max(segment1[:, None, 0], segment2[None, :, 0]))
        union = (segment1[:, None, 1] - segment1[:, None, 0]) + (segment2[None, :, 1] - segment2[None, :, 0]) - inter
        iou = inter / (union + 1e-6)
    elif segment1.ndim == 3 and segment2.ndim == 3:
        inter = torch.max(torch.zeros(1, device=segment1.device),
                          torch.min(segment1[:, :, None, 1], segment2[:, None, :, 1]) - torch.max(segment1[:, :, None, 0],
                                                                                                  segment2[:, None, :, 0]))
        union = (segment1[:, :, None, 1] - segment1[:, :, None, 0]) + (segment2[:, None, :, 1] - segment2[:, None, :, 0]) - inter
        iou = inter / (union + 1e-6)
    else:
        raise Exception('not implement for this shape, please check.')
    return iou


def segment_giou(segment1, segment2):
    assert (segment1[..., 1] >= segment1[..., 0]).all()
    assert (segment2[..., 1] >= segment2[..., 0]).all()

    if segment1.ndim == 2 and segment2.ndim == 2:
        inter = torch.max(torch.zeros(1, device=segment1.device),
                          torch.min(segment1[:, None, 1], segment2[None, :, 1]) - torch.max(segment1[:, None, 0], segment2[None, :, 0]))
        union = (segment1[:, None, 1] - segment1[:, None, 0]) + (segment2[None, :, 1] - segment2[None, :, 0]) - inter
        iou = inter / (union + 1e-6)

        area = torch.max(segment1[:, None, 1], segment2[None, :, 1]) - torch.min(segment1[:, None, 0], segment2[None, :, 0])
        giou = iou - (area - union) / (area + 1e-6)


    elif segment1.ndim == 3 and segment2.ndim == 3:
        inter = torch.max(torch.zeros(1, device=segment1.device),
                          torch.min(segment1[:, :, None, 1], segment2[:, None, :, 1]) - torch.max(segment1[:, :, None, 0],
                                                                                                  segment2[:, None, :, 0]))
        union = (segment1[:, :, None, 1] - segment1[:, :, None, 0]) + (segment2[:, None, :, 1] - segment2[:, None, :, 0]) - inter + 1e-5
        iou = inter / (union + 1e-6)

        area = torch.max(segment1[:, :, None, 1], segment2[:, None, :, 1]) - torch.min(segment1[:, :, None, 0], segment2[:, None, :, 0])
        giou = iou - (area - union) / (area + 1e-6)
    else:
        raise Exception('not implement for this shape, please check.')
    return giou


def ml2se(segment):
    mid = segment[..., 0:1]
    len = segment[..., 1:]
    start = mid - 0.5 * len
    end = mid + 0.5 * len
    start = torch.clamp(start, min=0, max=1)
    end = torch.clamp(end, min=0, max=1)
    return torch.cat([start, end], dim=-1)


def se2ml(segment):
    start = segment[:, :1]
    end = segment[:, 1:]
    mid = (start + end) * 0.5
    length = end - start
    return torch.cat([mid, length], dim=-1)


def preprocess_groundtruth(gt, original_len=None, to_tensor=False, device=None):
    # prepare the groudtruth format for HungarianMatcher

    gt_out = []
    if original_len is None:
        original_len = np.ones(len(gt))
    for v_vt, gt_len in zip(gt, original_len):
        annotation = {}
        # convert the gt type from mid-length to start-end
        v_gt_start_end = mid_dis2start_end(v_vt[:, 1:])
        v_gt_start = v_gt_start_end[:, 0, None]
        v_gt_start[v_gt_start < 0] = 0.
        v_gt_end = v_gt_start_end[:, 1, None]
        v_gt_end[v_gt_end >= 1] = 1.
        # set one video's gt label
        gt_data = np.concatenate([v_gt_start, v_gt_end], axis=-1)
        if to_tensor:
            annotation["segments"] = torch.tensor(gt_data, device=device)
            annotation["labels"] = torch.tensor((v_vt[:, 0]), device=device)
            annotation["length"] = torch.tensor((gt_len), device=device)
        else:
            annotation["segments"] = gt_data
            annotation["labels"] = (v_vt[:, 0])
            annotation["length"] = (gt_len)
        gt_out.append(annotation)
    return gt_out
