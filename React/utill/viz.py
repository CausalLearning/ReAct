import itertools
import os
import re
from textwrap import wrap

import PIL.Image as Image
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.compat.v1 as tf
import torch
from matplotlib.collections import LineCollection
from sklearn.manifold import TSNE
from torchvision import transforms
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from sklearn.metrics import confusion_matrix

tf.disable_v2_behavior()

color_table = np.array([[i, j, k] for i in np.linspace(0.2, 0.8, 4) for j in np.linspace(0.2, 0.8, 4) for k in np.linspace(0.2, 0.8, 4)])
matplotlib.use('Agg')


def center_spines(ax=None, centerx=0, centery=0, autoscale=True, scale=1):
    """Centers the axis spines at <centerx, centery> on the axis "ax", and
    places arrows at the end of the axis spines."""
    if ax is None:
        ax = plt.gca()

    if autoscale:
        plt.autoscale(True, tight=True)
    else:
        plt.autoscale(False)
        plt.xlim(-scale, scale)
        plt.ylim(-scale, scale)

    # Set the axis's spines to be centered at the given point
    # (Setting all 4 spines so that the tick marks go in both directions)
    ax.spines['left'].set_position(('data', centerx))
    ax.spines['bottom'].set_position(('data', centery))
    ax.spines['right'].set_position(('data', centerx - scale))
    ax.spines['top'].set_position(('data', centery - scale))

    # Hide the line (but not ticks) for "extra" spines
    for side in ['right', 'top']:
        ax.spines[side].set_color('none')


def embeddingto2d(data):
    print("start embedding...")
    # data shape (n, dim)
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    out = tsne.fit_transform(data)
    print("embedding finished.")
    return out


def vis_2d_scatter(x, y, label=None, Summarywritter=None, name='', step=-1):
    fig = plt.figure(0)
    center_spines()
    if label is None:
        plt.scatter(x, y, s=1, c='black', alpha=1)
    else:
        plt.scatter(x, y, s=1, c=color_table[label], alpha=1)

    if Summarywritter:
        Summarywritter.add_figure(name, fig, step)
        plt.clf()
    else:
        return fig


def vis_1d_box(all_data, Summarywritter=None, name='', step=-1):
    fig = plt.figure(0)
    # all_data: 所有一维data的list
    plt.boxplot(all_data, patch_artist=True, vert=True, notch=True)
    if Summarywritter:
        Summarywritter.add_figure(name, fig, step)
        plt.clf()
    else:
        return fig


def vis_feature_map(feature_map, Summarywritter=None, name='', step=-1):
    # feature_map shape: [channel, size, size]

    feature_map = feature_map.view(1, feature_map.shape[0], feature_map.shape[1], feature_map.shape[2])  # (1,64,55,55)
    upsample = torch.nn.UpsamplingBilinear2d(size=(256, 256))  # 这里进行调整大小
    feature_map = upsample(feature_map)
    feature_map = feature_map.view(feature_map.shape[1], feature_map.shape[2], feature_map.shape[3])

    feature_map_num = feature_map.shape[0]  # 返回通道数
    row_num = np.ceil(np.sqrt(feature_map_num))  # 8
    fig = plt.figure(0)
    for index in range(1, feature_map_num + 1):  # 通过遍历的方式，将64个通道的tensor拿出

        plt.subplot(row_num, row_num, index)
        # plt.imshow(feature_map[index - 1], cmap='gray')  # feature_map[0].shape=torch.Size([55, 55])
        # 将上行代码替换成，可显示彩色
        plt.imshow(transforms.ToPILImage()(feature_map[index - 1]))  # feature_map[0].shape=torch.Size([55, 55])
        plt.axis('off')

    # fig.show()
    if Summarywritter:
        Summarywritter.add_figure(name, fig, step)
        plt.clf()
    else:
        return fig


def reshape_transform(tensor, height=7, width=7):
    result = tensor.reshape(tensor.size(0),
                            height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def vis_CAM(model, target_layer, input, target_category=None, method=GradCAM, reshape_transform=None, merge_with_origin_image=False, rgb_image=None, use_cuda=True):
    # target_layers 一个列表
    # input 带batch维度
    # rgb_image 和 grayscale_cam都是numpy数组（opencv）
    # 常见的可视化层：
    # Resnet18 and 50: model.layer4[-1]
    # VGG and densenet161: model.features[-1]
    # mnasnet1_0: model.layers[-1]
    # ViT: model.blocks[-1].norm1
    # SwinT: model.layers[-1].blocks[-1].norm1

    cam = method(model=model, target_layer=target_layer, use_cuda=use_cuda, reshape_transform=reshape_transform)
    if input.shape[0] > 1:
        cam.batch_size = input.shape[0]
    grayscale_cam = cam(input_tensor=input, target_category=target_category)

    if merge_with_origin_image:
        grayscale_cam = grayscale_cam[0]
        vis = show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True)
        return vis
    else:
        return grayscale_cam


def viz_confusion_matrix(correct_labels, predict_labels, labels, normalize=False, Summarywritter=None, name='confusion_matrix', step=-1):
    '''
    Parameters:
        correct_labels                  : These are your true classification categories.
        predict_labels                  : These are you predicted classification categories
        labels                          : This is a lit of labels which will be used to display the axix labels
        title='Confusion matrix'        : Title for your matrix
        tensor_name = 'MyFigure/image'  : Name for the output summay tensor

    Returns:
        summary: TensorFlow summary

    Other itema to note:
        - Depending on the number of category and the data , you may have to modify the figzie, font sizes etc.
        - Currently, some of the ticks dont line up due to rotations.
    '''
    cm = confusion_matrix(correct_labels, predict_labels, labels=labels)
    if normalize:
        cm = cm.astype('float') * 10 / cm.sum(axis=1)[:, np.newaxis]
        cm = np.nan_to_num(cm, copy=True)
        cm = cm.astype('int')

    np.set_printoptions(precision=2)
    ###fig, ax = matplotlib.figure.Figure()

    fig = plt.Figure(figsize=(7, 7), dpi=320, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(cm, cmap='Oranges')

    classes = [re.sub(r'([a-z](?=[A-Z])|[A-Z](?=[A-Z][a-z]))', r'\1 ', x) for x in labels]
    classes = ['\n'.join(wrap(l, 40)) for l in classes]

    tick_marks = np.arange(len(classes))

    ax.set_xlabel('Predicted', fontsize=7)
    ax.set_xticks(tick_marks)
    c = ax.set_xticklabels(classes, fontsize=4, rotation=-90, ha='center')
    ax.xaxis.set_label_position('bottom')
    ax.xaxis.tick_bottom()

    ax.set_ylabel('True Label', fontsize=7)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, fontsize=4, va='center')
    ax.yaxis.set_label_position('left')
    ax.yaxis.tick_left()

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], 'd') if cm[i, j] != 0 else '.', horizontalalignment="center", fontsize=6, verticalalignment='center', color="black")
    fig.set_tight_layout(True)

    if Summarywritter:
        Summarywritter.add_figure(name, fig, step)
        plt.clf()
    else:
        return fig


def plot_segments(segments, labels, class_num, fig=None, ax=None):
    # segments:  [nedges, start/stop (2), xy (2)]
    # an array of xy values for each line to draw, with dimensions

    if fig is None:
        fig = plt.figure(0)
        ax = fig.gca()
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.5, 5)
        ax.set_yticks([])

    if class_num:
        labels = labels / class_num
        colors = plt.cm.jet(labels)
    else:
        # LineCollection wants a sequence of RGBA tuples, one for each line
        norm = plt.Normalize(labels.min(), labels.max())
        colors = plt.cm.jet(norm(labels))

    # we can now create a LineCollection from the xy and color values for each line
    lc = LineCollection(segments, colors=colors, linewidths=2, antialiased=True)
    ax.add_collection(lc)

    # we'll also plot some markers and numbers for the nodes
    ax.plot(segments[:, 0, 0], segments[:, 0, 1], 'ok', ms=1)
    ax.plot(segments[:, 1, 0], segments[:, 1, 1], 'ok', ms=1)

    # label with number
    for ni in range(segments.shape[0]):
        ax.annotate(str(ni), xy=segments[ni, 0, :], xytext=(-1, 3), textcoords='offset points', fontsize='xx-small')
        ax.annotate(str(ni), xy=segments[ni, 1, :], xytext=(-1, 3), textcoords='offset points', fontsize='xx-small')

    # to make a color bar, we first create a ScalarMappable, which will map the intensity values to the colormap scale
    # sm = plt.cm.ScalarMappable(norm, plt.cm.jet)
    # sm.set_array(z_connected)
    # cb = plt.colorbar(sm)
    # cb.set_label('Edge intensity')
    # ax.set_xlabel('Video time')
    # ax.set_ylabel('predict')

    return fig, ax


def viz_tad_segment(gt, pred, class_num, path="video.png"):
    # pred (#class, Mx3)    gt：map {segments: (K,2), labels: K}

    # plot ground truth
    # segments = gt["segments"] / bin_num  # normalized time to [0,1]
    segments = gt["segments"]  # normalized time to [0,1]
    segments = segments[:, :, None]
    y_axis = np.zeros_like(segments)  # set the ground truth laying in y=0
    segments = np.concatenate([segments, y_axis], axis=-1)
    labels = gt["labels"]
    fig, ax = plot_segments(segments, labels, class_num)

    # plot predicted results
    for i, each_class in enumerate(pred):
        if len(each_class) > 0:
            segments = each_class[:, :2]
            segments = segments[:, :, None]
            y_axis = np.ones_like(segments)  # set the predicted results laying in y = 1
            segments = np.concatenate([segments, y_axis], axis=-1)
            labels = np.ones(len(segments)) * i
            fig, ax = plot_segments(segments, labels, class_num, fig, ax)

    fig.savefig(path, dpi=400)
    fig.clf()
    return fig


def viz_tad_segment_all_dataset(gt, pred, class_num, out_dir='viz_out'):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    for i in range(len(pred)):
        viz_tad_segment(gt[i], pred[i], class_num, os.path.join(out_dir, '{}.png'.format(i)))


def viz_videos_with_test(frame_queue, threshold):
    text_info = {}
    while True:
        msg = 'Waiting for action ...'

        if len(result_queue) != 0:
            text_info = {}
            results = result_queue.popleft()
            for i, result in enumerate(results):
                selected_label, score = result
                if score < threshold:
                    break
                location = (0, 40 + i * 20)
                text = selected_label + ': ' + str(round(score, 2))
                text_info[location] = text
                cv2.putText(frame, text, location, FONTFACE, FONTSCALE,
                            FONTCOLOR, THICKNESS, LINETYPE)

        elif len(text_info) != 0:
            for location, text in text_info.items():
                cv2.putText(frame, text, location, FONTFACE, FONTSCALE,
                            FONTCOLOR, THICKNESS, LINETYPE)

        else:
            cv2.putText(frame, msg, (0, 40), FONTFACE, FONTSCALE, MSGCOLOR,
                        THICKNESS, LINETYPE)

        cv2.imshow('camera', frame)
        ch = cv2.waitKey(1)

        if ch == 27 or ch == ord('q') or ch == ord('Q'):
            break

        if drawing_fps > 0:
            # add a limiter for actual drawing fps <= drawing_fps
            sleep_time = 1 / drawing_fps - (time.time() - cur_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
            cur_time = time.time()


def save_images_from_event(event_path, tag, output_dir='./'):
    assert (os.path.isdir(output_dir))

    if isinstance(tag, str):
        tag = [tag]

    image_str = tf.placeholder(tf.string)
    im_tf = tf.image.decode_image(image_str)

    with tf.Session() as sess:
        count = 0
        for e in tf.train.summary_iterator(event_path):
            for v in e.summary.value:
                if v.tag in tag:
                    im = im_tf.eval({image_str: v.image.encoded_image_string})
                    output_fn = os.path.realpath('{}/image_{:05d}.png'.format(output_dir, count))
                    # print("Saving '{}'".format(output_fn))
                    im = Image.fromarray(im)
                    im.save(output_fn)
                    count += 1
