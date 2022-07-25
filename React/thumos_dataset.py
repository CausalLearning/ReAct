import copy
import os
import os.path
import time

import h5py
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from math import ceil
from numpy.random import randint

from React.utill.temporal_box_producess import softnms_v2
from mmaction.datasets.builder import DATASETS
from mmaction.datasets.pipelines import Compose
from tools.misc.evaluation import eval_map


def load_proposal_file(filename):
    lines = list(open(filename))
    from itertools import groupby
    groups = groupby(lines, lambda x: x.startswith('#'))

    info_list = [[x.strip() for x in list(g)] for k, g in groups if not k]

    def parse_group(info):
        offset = 0
        vid = info[offset]
        offset += 1

        n_frame = int(float(info[1]) * float(info[2]))
        n_gt = int(info[3])
        offset = 4

        gt_boxes = [x.split() for x in info[offset:offset + n_gt]]
        offset += n_gt
        # if n_gt == 0:
        #   offset += 1
        n_pr = int(info[offset])
        offset += 1
        pr_boxes = [x.split() for x in info[offset:offset + n_pr]]

        return vid, n_frame, gt_boxes, pr_boxes

    return [parse_group(l) for l in info_list]


class SegmentInstance:

    def __init__(self, start_frame, end_frame, video_frame_count,
                 fps=1, label=None,
                 best_iou=None, overlap_self=None):
        self.start_frame = start_frame
        self.end_frame = min(end_frame, video_frame_count)
        self._label = label
        self.fps = fps

        self.coverage = (end_frame - start_frame) / video_frame_count

        self.best_iou = best_iou
        self.overlap_self = overlap_self

    @property
    def start_time(self):
        return self.start_frame / self.fps

    @property
    def end_time(self):
        return self.end_frame / self.fps

    @property
    def label(self):
        return self._label if self._label is not None else -1


class VideoRecord:
    def __init__(self, prop_record):
        self._data = prop_record

        frame_count = int(self._data[1])

        # build instance record
        self.gt = [
            SegmentInstance(int(x[1]), int(x[2]), frame_count, label=int(x[0]), best_iou=1.0) for x in self._data[2]
            if int(x[2]) > int(x[1])
        ]

        self.gt = list(filter(lambda x: x.start_frame < frame_count, self.gt))

        self.proposals = [
            SegmentInstance(int(x[3]), int(x[4]), frame_count, label=int(x[0]),
                            best_iou=float(x[1]), overlap_self=float(x[2])) for x in self._data[3] if int(x[4]) > int(x[3])
        ]

        self.proposals = list(filter(lambda x: x.start_frame < frame_count, self.proposals))

    @property
    def id(self):
        return self._data[0].strip("\n").split("/")[-1]

    @property
    def num_frames(self):
        return int(self._data[1])


@DATASETS.register_module()
class ThumosDataset(data.Dataset):

    def __init__(self, prop_file, ft_path, pipeline, aux_ft_path=None, exclude_empty=True, epoch_multiplier=1, clip_len=128,
                 stride_rate=0.75, K=8, contrastive_epoch=0, test_mode=False, feature_type='TSN',
                 soft_nms_sigma=0.4,
                 soft_nms_threshold=1e-6,
                 ):

        self.ft_path = ft_path
        self.aux_ft_path = aux_ft_path
        self.prop_file = prop_file
        self.test_mode = test_mode
        self.clip_len = clip_len
        self.stride_rate = stride_rate
        self.feature_type = feature_type
        self.K = K
        self.contrastive_epoch = contrastive_epoch
        self.provide_contrastive_data = True if contrastive_epoch > 0 else False
        self.soft_nms_sigma = soft_nms_sigma
        self.soft_nms_threshold = soft_nms_threshold

        self.exclude_empty = exclude_empty
        self.epoch_multiplier = epoch_multiplier

        self.best = 0.

        self.pipeline = Compose(pipeline)

        parse_time = time.time()
        self._parse_data_list()

        if feature_type == 'TSN':
            self.feature_rgb = h5py.File(ft_path, 'r')
            if aux_ft_path is not None:
                self.feature_flow = h5py.File(aux_ft_path, 'r')

        if not test_mode:
            self.training_clip_list, self.cls_list = self.prepare_training_clip()
            self.real_len = len(self.training_clip_list)
        else:
            self.real_len = len(self.video_list)

        print("File parsed. Time:{:.2f}".format(time.time() - parse_time))

    def _parse_data_list(self):
        prop_info = load_proposal_file(self.prop_file)

        self.video_list = [VideoRecord(p) for p in prop_info]

        if self.exclude_empty:
            self.video_list = list(filter(lambda x: len(x.gt) > 0, self.video_list))

        # todo trick: sort for test(train, no used) dataset, then data of a batch have similiar len, which help reduce the testing time
        self.video_list = sorted(self.video_list, key=lambda x: x.num_frames)

        self.video_dict = {v.id: v for v in self.video_list}
        self.gt_list = self.get_all_gt()

    def get_all_gt(self):
        gt_list = []
        for video in self.video_list:
            gts = [[x.label - 1, (x.start_frame + x.end_frame) / (2 * video.num_frames), (x.end_frame - x.start_frame) / video.num_frames]
                   for x in video.gt]
            gt_list.append(sorted(gts, key=lambda x: x[1] - 0.5 * x[2]))  # sort gt by time
        return gt_list

    def keep_gt_with_thershold(self, gt, start_ratio, end_ratio, thershold):
        gt_start = gt[:, 1] - 0.5 * gt[:, 2]
        gt_end = gt[:, 1] + 0.5 * gt[:, 2]

        inter = np.maximum(0, np.minimum(gt_end, end_ratio) - np.maximum(gt_start, start_ratio))
        union = gt_end - gt_start

        gt_iou = inter / union

        idx = gt_iou >= thershold
        return gt[idx]

    def rescale_gt(self, gt, start_ratio, origin_len, new_len):
        gt[:, 1] = (gt[:, 1] - start_ratio) * (origin_len / new_len)
        gt[:, 2] = gt[:, 2] * (origin_len / new_len)

        return gt

    def prepare_training_clip(self):
        all_clip_list = []
        cls_list = [[] for _ in range(20)]
        for v_idx in range(len(self.video_list)):

            video = self.video_list[v_idx]
            gt = self.gt_list[v_idx]
            gt = np.array(gt)

            vid_full_name = video.id
            vid_num_frames = video.num_frames
            vid = vid_full_name.split('/')[-1]

            if self.feature_type == 'TSN':
                ft_tensor = self.feature_rgb[vid][:][::5]
                ft_tensor = torch.from_numpy(ft_tensor)
                if self.aux_ft_path:
                    aux_ft_tensor = self.feature_flow[vid][:][::5]
                    aux_ft_tensor = torch.from_numpy(aux_ft_tensor)
                    ft_tensor = torch.cat([ft_tensor, aux_ft_tensor], dim=-1)
            else:
                ft_tensor = torch.load(os.path.join(self.ft_path, vid)).float()
                if self.aux_ft_path:
                    aux_ft_tensor = torch.load(os.path.join(self.aux_ft_path, vid)).float()
                    if len(ft_tensor) > len(aux_ft_tensor):
                        ft_tensor = ft_tensor[:len(aux_ft_tensor)]
                    ft_tensor = torch.cat([ft_tensor, aux_ft_tensor], dim=-1)

            snippet_num = len(ft_tensor)

            clips_num = ceil((snippet_num - self.clip_len) / (self.clip_len * self.stride_rate)) + 1
            clips_ratio = self.clip_len / snippet_num

            if clips_num <= 1:
                result = {}
                result["raw_feature"] = ft_tensor  # (clip_len, feature_dim)
                result["gt_bbox"] = gt  # (gt_num, 3)
                result['snippet_num'] = snippet_num  # the snippet number of the clip
                result['video_name'] = vid
                result['origin_snippet_num'] = snippet_num  # the snippet number of the whole video
                v_cls_list = []
                for each_labels in gt:
                    cls = int(each_labels[0])
                    if cls not in v_cls_list:
                        v_cls_list.append(cls)
                        cls_list[cls].append(len(all_clip_list))
                all_clip_list.append(result)
                continue

            # sample all data for train
            for window_num in range(clips_num):

                start_snippet = int(self.clip_len * self.stride_rate) * window_num

                # finding matched gt
                start_ratio = (clips_ratio * self.stride_rate) * window_num
                end_ratio = start_ratio + clips_ratio

                window_feature = ft_tensor[start_snippet:start_snippet + self.clip_len, :]

                # find the matching gt in the windows
                selected_gt = self.keep_gt_with_thershold(gt, start_ratio, end_ratio, thershold=0.75)

                if len(selected_gt) == 0:
                    continue

                result = {}
                # selected_gt = gt[index]
                clip_gt = self.rescale_gt(selected_gt, start_ratio, snippet_num, len(window_feature))  # gt_num_i, 3

                result['raw_feature'] = window_feature
                result["snippet_num"] = len(window_feature)  # the snippet number of the clip
                result["gt_bbox"] = clip_gt  # (gt_num, 3)
                result['video_name'] = vid
                result['origin_snippet_num'] = snippet_num  # the snippet number of the whole video
                v_cls_list = []
                for each_labels in clip_gt:
                    cls = int(each_labels[0])
                    if cls not in v_cls_list:
                        v_cls_list.append(cls)
                        cls_list[cls].append(len(all_clip_list))
                all_clip_list.append(result)

        return all_clip_list, cls_list

    def prepare_testing_clip(self, result):
        features = result['raw_feature']
        snippet_num = len(features)
        gt = result['gt_bbox']

        clips_num = ceil((snippet_num - self.clip_len) / (self.clip_len * self.stride_rate)) + 1
        clips_ratio = self.clip_len / snippet_num

        clips_list = []
        gt_for_each_clips = []

        if clips_num <= 1:
            result["raw_feature"] = [features]  # 1 x clip_len x feature_dim
            result["gt_bbox"] = [gt]  # 1 x gt_num x 3
            result['snippet_num'] = torch.tensor([len(features)])
            return result

        snippet_num_list = []
        # sample all data for test
        for window_num in range(clips_num):
            start_snippet = int(self.clip_len * self.stride_rate) * window_num
            clips_list.append(features[start_snippet:start_snippet + self.clip_len, :])
            snippet_num_list.append(len(clips_list[-1]))

            # finding matched gt
            start_ratio = (clips_ratio * self.stride_rate) * window_num
            end_ratio = start_ratio + clips_ratio

            # find the matching gt in the windows
            index = np.logical_and((gt[:, 1] - gt[:, 2] / 2) >= start_ratio, (gt[:, 1] + gt[:, 2] / 2) <= end_ratio)

            clip_gt = self.rescale_gt(gt[index], start_ratio, snippet_num, len(clips_list[-1]))  # gt_num_i, 3
            gt_for_each_clips.append(clip_gt)

        result["raw_feature"] = clips_list  # clip_num, clip_len, feature_dim
        result["gt_bbox"] = gt_for_each_clips  # list: clip_num x [gt_num_i , 3]
        result["snippet_num"] = torch.tensor(snippet_num_list)
        return result

    def sample_by_class(self, class_id):
        instance_num = len(self.cls_list[class_id])
        sample_instance_id = np.random.choice(instance_num, 1)[0]
        sample = self.cls_list[class_id][sample_instance_id]

        return sample

    def sample_positive(self, class_id):
        instance = self.sample_by_class(class_id)
        pair_instances = self.training_clip_list[instance]
        while True:
            sample_segment_id = np.random.choice(len(pair_instances["gt_bbox"]), 1)[0]
            sample_segment = pair_instances["gt_bbox"][sample_segment_id]
            if sample_segment[0] == class_id:
                break

        sample_segment = np.array([sample_segment[1] - 0.5 * sample_segment[2], sample_segment[1] + 0.5 * sample_segment[2]])
        pair_feat = pair_instances["raw_feature"]
        return pair_feat, sample_segment

    def sample_same_class_negative(self, gt, k=4):
        gt_mid = gt[1]
        gt_len = gt[2]
        gt_start = gt_mid - 0.5 * gt_len
        gt_end = gt_mid + 0.5 * gt_len

        # sample negative pair
        if gt_len > 0.75:
            # sample inner negative pair
            random_shift_len = np.random.uniform(0.6, 0.9, k)
            inner_positive_sample_len = (1 - random_shift_len) * gt_len  # IoU< 0.4 will be treated as a negative sample
            random_start_neg = np.random.uniform(0, random_shift_len) * gt_len + gt_start
            random_end_neg = random_start_neg + inner_positive_sample_len
            candidated = np.stack([random_start_neg, random_end_neg], axis=-1)
        else:
            candidated = []
            # sample inner negative pair
            random_shift_len = np.random.uniform(0.5, 0.8, k)
            inner_positive_sample_len = (1 - random_shift_len) * gt_len  # IoU< 0.5 will be treated as a negative sample
            random_start_neg = np.random.uniform(0, random_shift_len) * gt_len + gt_start
            random_end_neg = random_start_neg + inner_positive_sample_len
            candidated.append(np.stack([random_start_neg, random_end_neg], axis=-1))

            # # sample left outter negative pair
            # random_inner_len_left = np.random.uniform(0.1, 0.5, k) * gt_len
            # random_out_len_left = np.random.uniform(0.3, 0.5, k) * gt_len
            #
            # left_start = gt_start - random_out_len_left
            # left_end = gt_start + random_inner_len_left
            #
            # left_sample = np.stack([left_start, left_end], axis=-1)
            # left_sample = left_sample[left_sample[:, 0] >= 0]
            # if len(left_sample) > 0:
            #     candidated.append(left_sample)
            #
            # # sample right outter negative pair
            # random_inner_len_right = np.random.uniform(0.1, 0.5, k) * gt_len
            # random_out_len_right = np.random.uniform(0.3, 0.5, k) * gt_len
            #
            # right_start = gt_end - random_inner_len_right
            # right_end = gt_end + random_out_len_right
            #
            # right_sample = np.stack([right_start, right_end], axis=-1)
            # right_sample = right_sample[right_sample[:, 0] <= 1]
            #
            # if len(right_sample) > 0:
            #     candidated.append(right_sample)

            candidated = np.concatenate(candidated, axis=0)
            choice_idx = np.random.choice(list(range(len(candidated))), k, replace=False)
            candidated = candidated[choice_idx]

        return candidated

    def sample_different_class_negative(self, class_id):
        while True:
            sample_class_id = np.random.randint(0, 20)
            if sample_class_id == class_id:
                continue
            else:
                break

        instance = self.sample_by_class(sample_class_id)
        pair_instances = self.training_clip_list[instance]
        while True:
            sample_segment_id = np.random.choice(len(pair_instances["gt_bbox"]), 1)[0]
            sample_segment = pair_instances["gt_bbox"][sample_segment_id]
            if sample_segment[0] != class_id:
                break

        sample_segment = np.array([sample_segment[1] - 0.5 * sample_segment[2], sample_segment[1] + 0.5 * sample_segment[2]])
        pair_feat = pair_instances["raw_feature"]
        return pair_feat, sample_segment

    def sample_pair(self, gt, gt_len):
        class_id = int(gt[0])

        # sample positive pair
        pos_feat, pos_sample_segment = self.sample_positive(class_id)
        # rescale the segment coords because we will padding at the end of the sequence later.
        pos_sample_segment = pos_sample_segment * (len(pos_feat) / self.clip_len)

        # sample negative pair
        neg_feat, neg_sample_segment = self.sample_different_class_negative(class_id)
        neg_sample_segment = neg_sample_segment * (len(neg_feat) / self.clip_len)

        # sample noise pair
        candidated_segments = self.sample_same_class_negative(gt, k=self.K)
        candidated_segments = candidated_segments * (gt_len / self.clip_len)

        return pos_feat, pos_sample_segment, neg_feat, neg_sample_segment, candidated_segments

    def dump_results(self, result, out=None):
        return self.evaluate(result)

    def evaluate(self, result, logger=None, metrics=None):
        result = pd.DataFrame(result).values
        result_pred = result[:, 0]
        result_gt = result[:, 1]

        softnms_v2(result_pred, sigma=self.soft_nms_sigma, top_k=1000, score_threshold=self.soft_nms_threshold)
        result_pred = copy.deepcopy(result_pred)
        thershold = np.linspace(0.3, 0.7, 5)
        iou_out = []
        for each in thershold:
            mean_ap, eval_results = eval_map(result_pred, result_gt, iou_thr=each, mode='anet', nproc=8)
            iou_out.append(np.round(mean_ap * 100, decimals=1))

        print("\nmAP for 0.3-0.7", iou_out)
        mean_map = np.round(np.mean(iou_out), decimals=1)
        print("Average mAP", mean_map)

        if mean_map > self.best:
            self.best = mean_map

        print("best mAP: ", self.best)
        return {'mAP': mean_map}

    def __getitem__(self, index):
        real_index = index % self.real_len

        # if training, get training clip (any clips from any videos)
        if not self.test_mode:
            result = self.training_clip_list[real_index]

            if self.provide_contrastive_data:
                # preparing pair
                gt = result["gt_bbox"]
                gt_len = len(gt)
                sample_gt = np.random.choice(gt_len, 1)[0]
                sample_gt = gt[sample_gt]
                pos_feat, pos_sample_segment, neg_feat, neg_sample_segment, candidated_segments = self.sample_pair(sample_gt, len(
                    result['raw_feature']))
                result['sample_gt'] = np.array([sample_gt[1] - 0.5 * sample_gt[2], sample_gt[1] + 0.5 * sample_gt[2]])
                result['pos_feat'] = pos_feat
                result['pos_sample_segment'] = pos_sample_segment
                result['neg_feat'] = neg_feat
                result['neg_sample_segment'] = neg_sample_segment
                result['candidated_segments'] = candidated_segments
            else:
                null_data = np.zeros(1)
                result['sample_gt'] = null_data
                result['pos_feat'] = null_data
                result['pos_sample_segment'] = null_data
                result['neg_feat'] = null_data
                result['neg_sample_segment'] = null_data
                result['candidated_segments'] = null_data

            return self.pipeline(result)

        # get testing data (all clips for a videos)
        video = self.video_list[real_index]

        vid_full_name = video.id
        vid_num_frames = video.num_frames
        vid = vid_full_name.split('/')[-1]

        if self.feature_type == 'TSN':
            ft_tensor = self.feature_rgb[vid][:][::5]
            ft_tensor = torch.from_numpy(ft_tensor)
            if self.aux_ft_path:
                aux_ft_tensor = self.feature_flow[vid][:][::5]
                aux_ft_tensor = torch.from_numpy(aux_ft_tensor)
                ft_tensor = torch.cat([ft_tensor, aux_ft_tensor], dim=-1)
        else:
            raise Exception("Not implement")

        gt = self.gt_list[real_index]

        result = {}
        result['video_name'] = vid
        result['origin_snippet_num'] = len(ft_tensor)  # The snippet number of the whole video
        result['snippet_num'] = len(ft_tensor)  # The snippet number of the clip
        result['raw_feature'] = ft_tensor
        result['gt_bbox'] = np.array(gt)
        result['video_gt_box'] = np.array(gt)

        result = self.prepare_testing_clip(result)

        return self.pipeline(result)

    def __len__(self):
        return self.real_len * self.epoch_multiplier
