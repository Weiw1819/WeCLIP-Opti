import os
import torch
import torch.nn.functional as F
from lxml import etree
from clip.utils import parse_xml_to_dict, scoremap2bbox
from clip.clip_text import class_names, new_class_names, class_names_coco, new_class_names_coco
from tqdm import tqdm
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from pytorch_grad_cam.utils.image import scale_cam_image
import cv2
import numpy as np


class ClipOutputTarget:
    def __init__(self, category):
        self.category = category
    def __call__(self, model_output):
        if len(model_output.shape) == 1:
            return model_output[self.category]
        return model_output[:, self.category]




def generate_clip_fts(image, model, require_all_fts=True):
    model = model.cuda()

    if len(image.shape) == 3:
        image = image.unsqueeze(0)
    h, w = image.shape[-2], image.shape[-1]
    image = image.cuda()
    
    image_features_all, attn_weight_list = model.encode_image(image, h, w, require_all_fts=require_all_fts)
        
    return image_features_all, attn_weight_list


def generate_trans_mat(aff_mask, attn_weight, grayscale_cam):
    aff_mask = aff_mask.view(1,grayscale_cam.shape[0] * grayscale_cam.shape[1])
    aff_mat = attn_weight

    trans_mat = aff_mat / torch.sum(aff_mat, dim=0, keepdim=True)
    trans_mat = trans_mat / torch.sum(trans_mat, dim=1, keepdim=True)

    for _ in range(2):
        trans_mat = trans_mat / torch.sum(trans_mat, dim=0, keepdim=True)
        trans_mat = trans_mat / torch.sum(trans_mat, dim=1, keepdim=True)
    trans_mat = (trans_mat + trans_mat.transpose(1, 0)) / 2

    for _ in range(1):
        trans_mat = torch.matmul(trans_mat, trans_mat)

    trans_mat = trans_mat * aff_mask
    
    return trans_mat


def compute_trans_mat(attn_weight):
    aff_mat = attn_weight

    trans_mat = aff_mat / torch.sum(aff_mat, dim=0, keepdim=True)
    trans_mat = trans_mat / torch.sum(trans_mat, dim=1, keepdim=True)

    for _ in range(2):
        trans_mat = trans_mat / torch.sum(trans_mat, dim=0, keepdim=True)
        trans_mat = trans_mat / torch.sum(trans_mat, dim=1, keepdim=True)
    trans_mat = (trans_mat + trans_mat.transpose(1, 0)) / 2

    for _ in range(1):
        trans_mat = torch.matmul(trans_mat, trans_mat)

    trans_mat = trans_mat

    return trans_mat


def generate_trans_mat_seg(aff_mask, attn_weight, grayscale_cam):
    aff_mask = aff_mask.view(1,grayscale_cam.shape[0] * grayscale_cam.shape[1])
    aff_mat = attn_weight

    trans_mat = aff_mat / torch.sum(aff_mat, dim=0, keepdim=True)
    trans_mat = trans_mat / torch.sum(trans_mat, dim=1, keepdim=True)

    for _ in range(2):
        trans_mat = trans_mat / torch.sum(trans_mat, dim=0, keepdim=True)
        trans_mat = trans_mat / torch.sum(trans_mat, dim=1, keepdim=True)
    trans_mat = (trans_mat + trans_mat.transpose(1, 0)) / 2

    for _ in range(1):
        trans_mat = torch.matmul(trans_mat, trans_mat)

    trans_mat = trans_mat * aff_mask
    
    return trans_mat





def perform_single_voc_cam(img_path, image, image_features, attn_weight_list, seg_attn, bg_text_features,
                       fg_text_features, cam, mode='train', require_seg_trans=False,
                       label_id_list=None):
    bg_text_features = bg_text_features.cuda()
    fg_text_features = fg_text_features.cuda()

    image = image.unsqueeze(0)
    h, w = image.shape[-2], image.shape[-1]

    if label_id_list is None:
        # 验证阶段从原始标签读取当前图像包含的类别及原始尺寸。
        with Image.open(img_path) as ori_image:
            ori_image = np.asarray(ori_image)
        ori_height, ori_width = ori_image.shape[:2]
        label_id_list = (np.unique(ori_image) - 1).tolist()
        if 255 in label_id_list:
            label_id_list.remove(255)
        if 254 in label_id_list:
            label_id_list.remove(254)
    else:
        # 训练 DataLoader 已提供图像级类别标签，避免主进程再次读取 PNG。
        label_id_list = [int(label_id) for label_id in label_id_list]
        ori_height, ori_width = h, w


    label_list = []
    for lid in label_id_list:
        label_list.append(new_class_names[int(lid)])
    label_id_list = [int(lid) for lid in label_id_list]

    keys = []

    cam_refined_list = []

    bg_features_temp = bg_text_features.cuda()  # [bg_id_for_each_image[im_idx]].to(device_id)
    fg_features_temp = fg_text_features[label_id_list].cuda()
    text_features_temp = torch.cat([fg_features_temp, bg_features_temp], dim=0)
    input_tensor = [image_features, text_features_temp.cuda(), h, w]

    for idx, label in enumerate(label_list):
        label_index = new_class_names.index(label)
        keys.append(label_index)
        targets = [ClipOutputTarget(label_list.index(label))]
        grayscale_cam, logits_per_image, attn_weight_last = cam(input_tensor=input_tensor,
                                                                targets=targets,
                                                                target_size=None)  # (ori_width, ori_height))

        grayscale_cam = grayscale_cam[0, :]

        if idx == 0:
            if require_seg_trans == True:
                attn_weight = torch.cat([attn_weight_list, attn_weight_last], dim=0)
                attn_weight = attn_weight[:, 1:, 1:][-6:] #-8

                # attn_diff = torch.abs(seg_attn - attn_weight)
                attn_diff = seg_attn - attn_weight
                attn_diff = torch.sum(attn_diff.flatten(1), dim=1)
                diff_th = torch.mean(attn_diff)

                attn_mask = torch.zeros_like(attn_diff)
                attn_mask[attn_diff <= diff_th] = 1

                attn_mask = attn_mask.reshape(-1, 1, 1)
                attn_mask = attn_mask.expand_as(attn_weight)
                attn_weight = torch.sum(attn_mask*attn_weight, dim=0) / (torch.sum(attn_mask, dim=0)+1e-5)

                attn_weight = attn_weight.detach()
                attn_weight = attn_weight * seg_attn.squeeze(0).detach()
            else:
                attn_weight = torch.cat([attn_weight_list, attn_weight_last], dim=0)
                attn_weight = attn_weight[:, 1:, 1:][-8:]
                attn_weight = torch.mean(attn_weight, dim=0)  # (1, hw, hw)
                attn_weight = attn_weight.detach()
            _trans_mat = compute_trans_mat(attn_weight)
        _trans_mat = _trans_mat.float()

        box, cnt = scoremap2bbox(scoremap=grayscale_cam, threshold=0.4, multi_contour_eval=True)
        aff_mask = torch.zeros((grayscale_cam.shape[0], grayscale_cam.shape[1])).cuda()
        for i_ in range(cnt):
            x0_, y0_, x1_, y1_ = box[i_]
            aff_mask[y0_:y1_, x0_:x1_] = 1

        aff_mask = aff_mask.view(1, grayscale_cam.shape[0] * grayscale_cam.shape[1])
        trans_mat = _trans_mat*aff_mask

        cam_to_refine = torch.FloatTensor(grayscale_cam).cuda()
        cam_to_refine = cam_to_refine.view(-1, 1)

        cam_refined = torch.matmul(trans_mat, cam_to_refine).reshape(h // 16, w // 16)
        cam_refined_list.append(cam_refined)

    if mode == 'train':
        return cam_refined_list, keys, w, h
    else:
        return cam_refined_list, keys, ori_width, ori_height


def perform_batch_voc_cam(img_paths, images, image_features_batch,
                          attn_weight_batch, seg_attn_batch,
                          bg_text_features, fg_text_features, cam,
                          mode='train', require_seg_trans=False,
                          label_id_lists=None):
    """批量生成 VOC CAM，一次处理 batch 内所有图像-类别对。"""
    device = images.device
    batch_size, _, h, w = images.shape
    bg_text_features = bg_text_features.to(device)
    fg_text_features = fg_text_features.to(device)

    if label_id_lists is None:
        label_id_lists = [None] * batch_size

    resolved_label_ids = []
    output_sizes = []
    for img_path, label_ids in zip(img_paths, label_id_lists):
        if label_ids is None:
            with Image.open(img_path) as ori_image:
                ori_image = np.asarray(ori_image)
            ori_height, ori_width = ori_image.shape[:2]
            label_ids = (np.unique(ori_image) - 1).tolist()
            if 255 in label_ids:
                label_ids.remove(255)
            if 254 in label_ids:
                label_ids.remove(254)
        else:
            label_ids = [int(label_id) for label_id in label_ids]
            ori_height, ori_width = h, w

        if not label_ids:
            raise ValueError(f'图像 {img_path} 未包含任何前景类别')

        resolved_label_ids.append(label_ids)
        output_sizes.append(
            (w, h) if mode == 'train' else (ori_width, ori_height)
        )

    # 每张图像的候选文本为“当前前景类别 + 背景类别”。按前景类别数量
    # 分组后批量执行 Grad-CAM，可避免 FP16 padding mask 的无效梯度，
    # 同时将原来的“每个类别一次调用”压缩为“每种类别数量一次调用”。
    text_features_per_image = [
        torch.cat([fg_text_features[label_ids], bg_text_features], dim=0)
        for label_ids in resolved_label_ids
    ]
    image_groups = {}
    for image_index, label_ids in enumerate(resolved_label_ids):
        image_groups.setdefault(len(label_ids), []).append(image_index)

    grayscale_cams_by_image = [
        [None] * len(label_ids) for label_ids in resolved_label_ids
    ]
    attn_weight_last_by_image = [None] * batch_size

    for _, image_indices in image_groups.items():
        pair_image_features = []
        pair_text_features = []
        pair_mappings = []
        targets = []

        for image_index in image_indices:
            label_ids = resolved_label_ids[image_index]
            text_features = text_features_per_image[image_index]
            for local_class_index in range(len(label_ids)):
                pair_image_features.append(image_features_batch[image_index])
                pair_text_features.append(text_features)
                pair_mappings.append((image_index, local_class_index))
                targets.append(ClipOutputTarget(local_class_index))

        # image_features 的格式为 (token, batch, channel)。每个图像-类别对
        # 作为一个 batch 样本，同组仅执行一次最后层 forward/backward。
        input_tensor = [
            torch.cat(pair_image_features, dim=1),
            torch.stack(pair_text_features, dim=0),
            h,
            w,
        ]
        group_cams, _, group_attn_weights = cam(
            input_tensor=input_tensor,
            targets=targets,
            target_size=None,
        )

        for pair_index, (image_index, local_class_index) in enumerate(pair_mappings):
            grayscale_cams_by_image[image_index][local_class_index] = group_cams[pair_index]
            if attn_weight_last_by_image[image_index] is None:
                attn_weight_last_by_image[image_index] = group_attn_weights[
                    pair_index:pair_index + 1
                ]

    batch_results = []
    for image_index, label_ids in enumerate(resolved_label_ids):
        cam_refined_list = []

        # 同一图像复制出的类别对具有相同注意力，取第一个即可。
        attn_weight_last = attn_weight_last_by_image[image_index]
        if require_seg_trans:
            attn_weight = torch.cat(
                [attn_weight_batch[image_index], attn_weight_last], dim=0
            )
            attn_weight = attn_weight[:, 1:, 1:][-6:]
            seg_attn = seg_attn_batch[image_index:image_index + 1]

            attn_diff = seg_attn - attn_weight
            attn_diff = torch.sum(attn_diff.flatten(1), dim=1)
            diff_th = torch.mean(attn_diff)

            attn_mask = torch.zeros_like(attn_diff)
            attn_mask[attn_diff <= diff_th] = 1
            attn_mask = attn_mask.reshape(-1, 1, 1).expand_as(attn_weight)
            attn_weight = torch.sum(attn_mask * attn_weight, dim=0) / (
                torch.sum(attn_mask, dim=0) + 1e-5
            )
            attn_weight = attn_weight.detach()
            attn_weight = attn_weight * seg_attn.squeeze(0).detach()
        else:
            attn_weight = torch.cat(
                [attn_weight_batch[image_index], attn_weight_last], dim=0
            )
            attn_weight = attn_weight[:, 1:, 1:][-8:]
            attn_weight = torch.mean(attn_weight, dim=0).detach()

        trans_mat_base = compute_trans_mat(attn_weight).float()

        for local_class_index in range(len(label_ids)):
            grayscale_cam = grayscale_cams_by_image[
                image_index
            ][local_class_index]
            box, cnt = scoremap2bbox(
                scoremap=grayscale_cam,
                threshold=0.4,
                multi_contour_eval=True,
            )
            aff_mask = torch.zeros(
                grayscale_cam.shape, device=device, dtype=torch.float32
            )
            for box_index in range(cnt):
                x0, y0, x1, y1 = box[box_index]
                aff_mask[y0:y1, x0:x1] = 1

            trans_mat = trans_mat_base * aff_mask.view(1, -1)
            cam_to_refine = torch.as_tensor(
                grayscale_cam, device=device, dtype=torch.float32
            ).view(-1, 1)
            cam_refined = torch.matmul(
                trans_mat, cam_to_refine
            ).reshape(h // 16, w // 16)
            cam_refined_list.append(cam_refined)

        output_width, output_height = output_sizes[image_index]
        batch_results.append(
            (cam_refined_list, label_ids, output_width, output_height)
        )

    return batch_results




def generate_cam_label(cam_refined_list, keys, w, h):
    # 在 GPU 上一次性完成所有类别 CAM 的归一化与上采样，避免逐类别
    # GPU -> NumPy/OpenCV -> GPU 的往返传输和同步。
    refined_cams = torch.stack(cam_refined_list, dim=0).float().unsqueeze(1)
    refined_cams = refined_cams - refined_cams.amin(dim=(2, 3), keepdim=True)
    refined_cams = refined_cams / (
        refined_cams.amax(dim=(2, 3), keepdim=True) + 1e-7
    )
    refined_cams = F.interpolate(
        refined_cams,
        size=(h, w),
        mode='bilinear',
        align_corners=False,
    ).squeeze(1)

    return {
        'keys': np.asarray(keys, dtype=np.int64),
        'refined_cam': refined_cams,
    }




def perform_single_coco_cam(img_path, image, image_features, attn_weight_list, seg_attn, bg_text_features,
                        fg_text_features, cam, mode='train', require_all_fts=True, require_seg_trans=False):
    bg_text_features = bg_text_features.cuda()
    fg_text_features = fg_text_features.cuda()

    ori_image = Image.open(img_path)
    ori_height, ori_width = np.asarray(ori_image).shape[:2]
    label_id_list = np.unique(ori_image)
    label_id_list = (label_id_list-1).tolist()
    if 255 in label_id_list:
        label_id_list.remove(255)
    if 254 in label_id_list:
        label_id_list.remove(254)

    # print(label_id_list)
    label_list = []
    for lid in label_id_list:
        label_list.append(new_class_names_coco[int(lid)])
    label_id_list = [int(lid) for lid in label_id_list]

    image = image.unsqueeze(0)
    h, w = image.shape[-2], image.shape[-1]

    highres_cam_to_save = []
    keys = []

    cam_refined_list = []

    bg_features_temp = bg_text_features.cuda()  # [bg_id_for_each_image[im_idx]].to(device_id)
    fg_features_temp = fg_text_features[label_id_list].cuda()
    text_features_temp = torch.cat([fg_features_temp, bg_features_temp], dim=0)
    input_tensor = [image_features, text_features_temp.cuda(), h, w]

    for idx, label in enumerate(label_list):
        label_index = new_class_names_coco.index(label)
        keys.append(label_index)
        targets = [ClipOutputTarget(label_list.index(label))]
        grayscale_cam, logits_per_image, attn_weight_last = cam(input_tensor=input_tensor,
                                                                targets=targets,
                                                                target_size=None)  # (ori_width, ori_height))

        grayscale_cam = grayscale_cam[0, :]

        grayscale_cam_highres = cv2.resize(grayscale_cam, (w, h))
        highres_cam_to_save.append(torch.tensor(grayscale_cam_highres))

        # if idx == 0:
        #     attn_weight = torch.cat([attn_weight_list, attn_weight_last], dim=0)
        #     attn_weight = attn_weight[:, 1:, 1:][-8:]
        #     attn_weight = torch.mean(attn_weight, dim=0)  # (1, hw, hw)
        #     attn_weight = attn_weight.detach()
        #     if require_seg_trans == True:
        #         attn_weight = attn_weight * seg_attn.squeeze(0).detach()
        if idx == 0:
            if require_seg_trans == True:
                attn_weight = torch.cat([attn_weight_list, attn_weight_last], dim=0)
                attn_weight = attn_weight[:, 1:, 1:][-10:]  # -8

                # attn_diff = torch.abs(seg_attn - attn_weight)
                attn_diff = seg_attn - attn_weight
                attn_diff = torch.sum(attn_diff.flatten(1), dim=1)
                diff_th = torch.mean(attn_diff)

                attn_mask = torch.zeros_like(attn_diff)
                attn_mask[attn_diff <= diff_th] = 1

                attn_mask = attn_mask.reshape(-1, 1, 1)
                attn_mask = attn_mask.expand_as(attn_weight)
                attn_weight = torch.sum(attn_mask * attn_weight, dim=0) / (torch.sum(attn_mask, dim=0) + 1e-5)

                attn_weight = attn_weight.detach()
                attn_weight = attn_weight * seg_attn.squeeze(0).detach()
            else:
                attn_weight = torch.cat([attn_weight_list, attn_weight_last], dim=0)
                attn_weight = attn_weight[:, 1:, 1:][-8:]
                attn_weight = torch.mean(attn_weight, dim=0)  # (1, hw, hw)
                attn_weight = attn_weight.detach()
            _trans_mat = compute_trans_mat(attn_weight)
        _trans_mat = _trans_mat.float()

        box, cnt = scoremap2bbox(scoremap=grayscale_cam, threshold=0.7, multi_contour_eval=True)
        aff_mask = torch.zeros((grayscale_cam.shape[0], grayscale_cam.shape[1]))
        for i_ in range(cnt):
            x0_, y0_, x1_, y1_ = box[i_]
            aff_mask[y0_:y1_, x0_:x1_] = 1

        aff_mask = aff_mask.view(1, grayscale_cam.shape[0] * grayscale_cam.shape[1])
        trans_mat = _trans_mat.cuda() * aff_mask.cuda()

        cam_to_refine = torch.FloatTensor(grayscale_cam).cuda()
        cam_to_refine = cam_to_refine.view(-1, 1)

        cam_refined = torch.matmul(trans_mat, cam_to_refine).reshape(h // 16, w // 16)
        cam_refined_list.append(cam_refined)

    if mode == 'train':
        return cam_refined_list, keys, w, h
    else:
        return cam_refined_list, keys, ori_width, ori_height
