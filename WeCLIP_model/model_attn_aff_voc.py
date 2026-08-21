import torch
import torch.nn as nn
import torch.nn.functional as F
from .segformer_head import SegFormerHead
import numpy as np
import clip
from clip.clip_text import new_class_names, BACKGROUND_CATEGORY
from pytorch_grad_cam import GradCAM
from clip.clip_tool import generate_cam_label, generate_clip_fts, perform_batch_voc_cam
import os
from torchvision.transforms import Compose, Normalize
from .Decoder.TransDecoder import DecoderTransformer
from WeCLIP_model.PAR import PAR




def Normalize_clip():
    return Compose([
    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])


def reshape_transform(tensor, height=28, width=28):
    tensor = tensor.permute(1, 0, 2)
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result



def zeroshot_classifier(classnames, templates, model):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in classnames:
            texts = [template.format(classname) for template in templates] #format with class
            texts = clip.tokenize(texts).cuda() #tokenize
            class_embeddings = model.encode_text(texts) #embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights.t()


def _refine_cams(ref_mod, images, cams, valid_key):
    single_sample = images.ndim == 3
    if single_sample:
        images = images.unsqueeze(0)
    if cams.ndim == 3:
        cams = cams.unsqueeze(0)
    if valid_key.ndim == 1:
        valid_key = valid_key.unsqueeze(0)

    refined_cams = ref_mod(images.float(), cams.float())
    refined_label = refined_cams.argmax(dim=1)
    # 每个样本包含的有效类别不同，按样本将 CAM 通道映射到 VOC 类别编号。
    refined_label = torch.gather(
        valid_key,
        dim=1,
        index=refined_label.flatten(1),
    ).view_as(refined_label)

    return refined_label.squeeze(0) if single_sample else refined_label


class WeCLIP(nn.Module):
    def __init__(self, num_classes=None, clip_model=None, embedding_dim=256, in_channels=512, dataset_root_path=None, device='cuda'):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim

        self.encoder, _ = clip.load(clip_model, device=device)

        for name, param in self.encoder.named_parameters():
            if "11" not in name:
                param.requires_grad=False

        # for name, param in self.encoder.named_parameters():
        #     print(name, param.requires_grad)

        self.in_channels = in_channels

        self.decoder_fts_fuse = SegFormerHead(in_channels=self.in_channels,embedding_dim=self.embedding_dim,
                                              num_classes=self.num_classes, index=11)
        self.decoder = DecoderTransformer(width=self.embedding_dim, layers=3, heads=8, output_dim=self.num_classes)

        self.bg_text_features = zeroshot_classifier(BACKGROUND_CATEGORY, ['a clean origami {}.'], self.encoder)
        self.fg_text_features = zeroshot_classifier(new_class_names, ['a clean origami {}.'], self.encoder)

        self.target_layers = [self.encoder.visual.transformer.resblocks[-1].ln_1]
        self.grad_cam = GradCAM(model=self.encoder, target_layers=self.target_layers, reshape_transform=reshape_transform)
        self.root_path = os.path.join(dataset_root_path, 'SegmentationClassAug')
        self.cam_bg_thres = 1
        self.encoder.eval()
        self.par = PAR(num_iter=20, dilations=[1,2,4,8,12,24]).cuda()
        self.iter_num = 0
        self.require_all_fts = True


    def get_param_groups(self):

        param_groups = [[], [], [], []]  # backbone; backbone_norm; cls_head; seg_head;

        for param in list(self.decoder.parameters()):
            param_groups[3].append(param)
        for param in list(self.decoder_fts_fuse.parameters()):
            param_groups[3].append(param)

        return param_groups
    


    def forward(self, img, img_names='2007_000032', mode='train', cls_labels=None):
        """前向计算：生成分割预测结果，同时生成经过细化的 CAM 伪标签。"""
        cam_list = []
        # 获取输入图像的批大小、通道数以及空间尺寸。
        b, c, h, w = img.shape
        # CLIP 编码器只用于提取特征和计算 CAM，这里保持其评估模式。
        self.encoder.eval()
        # 记录当前迭代次数，用于控制 CAM 后处理策略。
        self.iter_num += 1

        # 提取 CLIP 的多层图像 token 特征和注意力权重。
        fts_all, attn_weight_list = generate_clip_fts(img, self.encoder, require_all_fts=True)

        # 将各层特征堆叠起来，形状通常为 (层数, token数, batch, 通道数)。
        fts_all_stack = torch.stack(fts_all, dim=0) # (11, hw, b, c)
        # 整理各层注意力权重，便于后续按样本取出对应的注意力图。
        attn_weight_stack = torch.stack(attn_weight_list, dim=0).permute(1, 0, 2, 3)
        if self.require_all_fts==True:
            # 只使用最后一层特征生成 CAM；调整维度后，第一维对应 batch。
            cam_fts_all = fts_all_stack[-1].unsqueeze(0).permute(2, 1, 0, 3) #(1, hw, 1, c)
        else:
            # 保留所有层的特征生成 CAM。
            cam_fts_all = fts_all_stack.permute(2, 1, 0, 3)

        # 去掉每层第一个 CLS token，仅保留空间图像 token。
        all_img_tokens = fts_all_stack[:, 1:, ...]
        img_tokens_channel = all_img_tokens.size(-1)
        # 将 token 维度还原为二维特征图，供 SegFormer 特征融合头使用。
        all_img_tokens = all_img_tokens.permute(0, 2, 3, 1)
        all_img_tokens = all_img_tokens.reshape(-1, b, img_tokens_channel, h//16, w //16) #(11, b, c, h, w)


        # 融合 CLIP 的多层特征，得到解码器所需的高层语义特征。
        fts = self.decoder_fts_fuse(all_img_tokens)
        # 保留一份特征副本，用于计算像素之间的自注意力关系。
        attn_fts = fts.clone()
        _, _, fts_h, fts_w = fts.shape
        
        # 解码得到最终的语义分割预测，以及解码器产生的注意力权重。
        seg, seg_attn_weight_list = self.decoder(fts)
        
        # 将特征展平为 (batch, 通道数, 像素数)，计算像素两两之间的相似度。
        f_b, f_c, f_h, f_w = attn_fts.shape
        attn_fts_flatten = attn_fts.reshape(f_b, f_c, f_h*f_w)
        attn_pred = attn_fts_flatten.transpose(2, 1).bmm(attn_fts_flatten)
        # 将相似度归一化到 0～1，作为后续 CAM 细化使用的分割注意力。
        attn_pred = torch.sigmoid(attn_pred)

        # 准备每张图像的标签路径和前景类别。训练集已有图像级标签，因此不再读 PNG。
        img_paths = [
            os.path.join(self.root_path, str(img_name) + '.png')
            for img_name in img_names
        ]
        label_id_lists = None
        if cls_labels is not None:
            label_id_lists = [
                torch.nonzero(cls_labels[i], as_tuple=False).flatten().tolist()
                for i in range(b)
            ]

        # 训练前期不进行分割变换；迭代足够多或验证阶段才启用该后处理。
        require_seg_trans = self.iter_num > 15000 or mode == 'val'

        # 将 batch 内所有“图像 × 前景类别”合并为一次 Grad-CAM forward/backward，
        # 减少大量串行的小型 CUDA 任务和 CPU/GPU 同步。
        batch_cam_results = perform_batch_voc_cam(
            img_paths=img_paths,
            images=img,
            image_features_batch=cam_fts_all,
            attn_weight_batch=attn_weight_stack,
            seg_attn_batch=attn_pred,
            bg_text_features=self.bg_text_features,
            fg_text_features=self.fg_text_features,
            cam=self.grad_cam,
            mode=mode,
            require_seg_trans=require_seg_trans,
            label_id_lists=label_id_lists,
        )

        for cam_refined_list, keys, w, h in batch_cam_results:


            # 将不同类别的 CAM 整理成统一格式，并恢复到原图空间尺寸。
            cam_dict = generate_cam_label(cam_refined_list, keys, w, h)
            
            # 取出已在 GPU 上完成归一化和上采样的前景 CAM。
            cams = cam_dict['refined_cam']

            # 由所有前景类别的最大响应估计背景分数，并将背景 CAM 拼到类别维最前面。
            bg_score = torch.pow(1 - torch.max(cams, dim=0, keepdims=True)[0], self.cam_bg_thres)
            cams = torch.cat([bg_score, cams], dim=0)
            
            # 建立 CAM 通道索引到 VOC 类别编号的映射，0 表示背景。
            valid_key = np.pad(cam_dict['keys'] + 1, (1, 0), mode='constant')
            valid_key = torch.from_numpy(valid_key).to(device=cams.device, dtype=torch.long)
            
            # 暂存当前样本，循环结束后统一执行 batch PAR。
            cam_list.append((cams, valid_key))

        # 不同图像包含的前景类别数可能不同，补零到当前 batch 的最大通道数。
        # PAR 对 batch 内图像并行细化，减少逐图执行产生的大量小型 CUDA kernel。
        max_cam_channels = max(cams.shape[0] for cams, _ in cam_list)
        cam_batch = img.new_zeros((b, max_cam_channels, h, w), dtype=torch.float32)
        valid_key_batch = torch.zeros(
            (b, max_cam_channels),
            device=img.device,
            dtype=torch.long,
        )
        for i, (cams, valid_key) in enumerate(cam_list):
            num_channels = cams.shape[0]
            cam_batch[i, :num_channels] = cams
            valid_key_batch[i, :num_channels] = valid_key

        # 使用 PAR（像素相似性传播）批量平滑 CAM，并得到离散伪标签。
        with torch.no_grad():
            all_cam_labels = _refine_cams(
                self.par,
                img,
                cam_batch,
                valid_key_batch,
            )

        # 返回分割预测、CAM 伪标签和像素级注意力矩阵。
        return seg, all_cam_labels, attn_pred

        
