from typing import Any, Dict
from torch.cuda.amp.autocast_mode import autocast
import torch
from mmcv.runner import auto_fp16, force_fp32
from torch import nn
from torch.nn import functional as F
from mmdet3d.models.heads.bbox.centerpoint import CenterHead
from mmdet3d.models.builder import (
    build_backbone,
    build_fuser,
    build_head,
    build_neck,
    build_vtransform,
)
from mmdet3d.ops import Voxelization, DynamicScatter
from mmdet3d.models import FUSIONMODELS
from mmdet.core import multi_apply
from .deformAttention import DFA
from .dcn import DeformableConv2d
from .base import Base3DFusionModel
from mmdet3d.models.utils.flops_counter import flops_counter
from thop import clever_format
from tools.visualize_feature import save_feature_to_img_cam,save_feature_to_img_lidarbranch
__all__ = ["BEVFusion"]


class DeformConv(nn.Module):
    def __init__(self,
                 deformable=False):
        super(DeformConv, self).__init__()

        # self.conv1 = nn.Conv2d(80, 32, kernel_size=3, stride=1, padding=1, bias=True)
        # self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True)
        # self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True)
        conv = nn.Conv2d if deformable == False else DeformableConv2d
        self.defconv1 = conv(80, 80, kernel_size=3, stride=1, padding=1, bias=True)
        self.defconv2 = conv(80, 80, kernel_size=3, stride=1, padding=1, bias=True)

        # self.pool = nn.MaxPool2d(2)
        # self.gap = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(32, 10)

    def forward(self, x):
        # x = torch.relu(self.conv1(x))
        # x = self.pool(x)  # [14, 14]
        # x = torch.relu(self.conv2(x))
        # x = self.pool(x)  # [7, 7]
        # x = torch.relu(self.conv3(x))
        # x = torch.relu(self.conv4(x))
        # x = torch.relu(self.conv5(x))
        # x = self.gap(x)
        # x = x.flatten(start_dim=1)
        # x = self.fc(x)
        x = torch.relu(self.defconv1(x))
        x = self.defconv2(x)
        return x
class QualityFocalLoss_no_reduction(nn.Module):
    '''
    input[B,M,C] not sigmoid
    target[B,M,C], sigmoid
    '''
    def __init__(self, beta = 2.0):

        super(QualityFocalLoss_no_reduction, self).__init__()
        self.beta = beta
    def forward(self, input: torch.Tensor, target: torch.Tensor, pos_normalizer=torch.tensor(1.0)):

        pred_sigmoid = torch.sigmoid(input)
        scale_factor = pred_sigmoid-target
        loss = F.binary_cross_entropy_with_logits(input, target, reduction='none')*(scale_factor.abs().pow(self.beta))
        loss /= torch.clamp(pos_normalizer, min=1.0)
        return loss

def unified_focal_loss(prob_volume, depth_values, interval, depth_gt, mask, weight, gamma, alpha):
    depth_gt_volume = depth_gt.unsqueeze(1).expand_as(depth_values)  # (b, d, h, w)

    gt_index_volume = ((depth_values <= depth_gt_volume) * ((depth_values + interval) > depth_gt_volume))

    gt_unity_index_volume = torch.zeros_like(prob_volume, requires_grad=False)
    gt_unity_index_volume[gt_index_volume] = 1.0 - (depth_gt_volume[gt_index_volume] - depth_values[gt_index_volume]) / interval

    gt_unity, _ = torch.max(gt_unity_index_volume, dim=1, keepdim=True)
    gt_unity = torch.where(gt_unity > 0.0, gt_unity, torch.ones_like(gt_unity))  # (b, 1, h, w)
    pos_weight = (sigmoid((gt_unity - prob_volume).abs() / gt_unity, base=5) - 0.5) * 4 + 1  # [1, 3]
    neg_weight = (sigmoid(prob_volume / gt_unity, base=5) - 0.5) * 2  # [0, 1]
    focal_weight = pos_weight.pow(gamma) * (gt_unity_index_volume > 0.0).float() + alpha * neg_weight.pow(gamma) * (
            gt_unity_index_volume <= 0.0).float()

    mask = mask.unsqueeze(1).expand_as(depth_values).float()
    loss = (F.binary_cross_entropy(prob_volume, gt_unity_index_volume, reduction="none") * focal_weight * mask).sum() / mask.sum()
    loss = loss * weight
    return loss


@FUSIONMODELS.register_module()
class BEVFusion(Base3DFusionModel):
    def __init__(
        self,
        encoders: Dict[str, Any],
        fuser: Dict[str, Any],
        fuser_student: Dict[str, Any],
        trans_student: Dict[str, Any],
        decoder: Dict[str, Any],
        heads: Dict[str, Any],
        **kwargs,
    ) -> None:
        super().__init__()

        self.encoders = nn.ModuleDict()
        if encoders.get("camera") is not None:
            self.encoders["camera"] = nn.ModuleDict(
                {
                    "backbone": build_backbone(encoders["camera"]["backbone"]),
                    "neck": build_neck(encoders["camera"]["neck"]),
                    "vtransform": build_vtransform(encoders["camera"]["vtransform"]),
                }
            )

        self.encoders_student = nn.ModuleDict()
        if encoders.get("camera") is not None:
            self.encoders_student["camera"] = nn.ModuleDict(
                {
                    "backbone": build_backbone(encoders["camera"]["backbone"]),
                    "neck": build_neck(encoders["camera"]["neck"]),
                    "vtransform": build_vtransform(encoders["camera"]["vtransform"]),
                    "vtransform2": build_vtransform(encoders["camera"]["vtransform"]),
                }
            )
        if encoders.get("lidar") is not None:
            if encoders["lidar"]["voxelize"].get("max_num_points", -1) > 0:
                voxelize_module = Voxelization(**encoders["lidar"]["voxelize"])
            else:
                voxelize_module = DynamicScatter(**encoders["lidar"]["voxelize"])
            self.encoders["lidar"] = nn.ModuleDict(
                {
                    "voxelize": voxelize_module,
                    "backbone": build_backbone(encoders["lidar"]["backbone"]),
                }
            )
            self.voxelize_reduce = encoders["lidar"].get("voxelize_reduce", True)

        if fuser is not None:
            self.fuser = build_fuser(fuser)
            self.fuser_student = build_fuser(fuser_student)
            self.trans_student = build_fuser(trans_student)
        else:
            self.fuser = None

        self.decoder = nn.ModuleDict(
            {
                "backbone": build_backbone(decoder["backbone"]),
                "neck": build_neck(decoder["neck"]),
            }
        )
        self.decoder_student = nn.ModuleDict(
            {
                "backbone": build_backbone(decoder["backbone"]),
                "neck": build_neck(decoder["neck"]),
            }
        )
        self.heads = nn.ModuleDict()
        for name in heads:
            if heads[name] is not None:
                self.heads[name] = build_head(heads[name])
        self.heads_student = nn.ModuleDict()
        for name in heads:
            if heads[name] is not None:
                self.heads_student[name] = build_head(heads[name])

        if "loss_scale" in kwargs:
            self.loss_scale = kwargs["loss_scale"]
        else:
            self.loss_scale = dict()
            for name in heads:
                if heads[name] is not None:
                    self.loss_scale[name] = 1.0
        self.feature_loss = nn.MSELoss(reduction='mean')
        self.fusion_feature = encoders["fusion_feature"]
        self.plidar_feature = encoders["plidar_feature"]
        self.plidar_mask = encoders["plidar_mask"]
        self.img_feature = encoders["img_feature"]
        self.sem_feature = encoders["sem_feature"]
        self.depth_feature = encoders["depth_feature"]
        self.depth_sup = encoders["depth_sup"]
        self.prediction_distill = encoders["prediction_distill"]
        self.bev_deformAttn = encoders["bev_DeformAttn"]
        self.bev_deform = encoders["bev_Deform"]
        self.dbound = encoders["camera"]["vtransform"]["dbound"]
        self.downsample_factor = 8
        self.depth_channels = int(
            (self.dbound[1] - self.dbound[0]) / self.dbound[2])

        self.d_model_bev = 80
        self.n_levels_bev = 1
        self.n_heads_bev = 8
        self.n_points_bev = 20
        self.bev_h = 180
        self.bev_w = 180
        self.num_att_bev = 1
        self.num_proj_bev = 1
        self.deformAttn = DFA(self.d_model_bev, self.bev_h, self.bev_w, self.num_att_bev, self.num_proj_bev, self.n_heads_bev, self.n_points_bev)
        # self.objectMask = CenterHead
        if self.bev_deform:
            self.deform = DeformConv(True)
        else:
            self.deform = DeformConv(False)


        self.init_weights()
        self.freeze = True
        if self.freeze:
            self.freeze_params()
    def init_weights(self) -> None:
        if "camera" in self.encoders:
            self.encoders["camera"]["backbone"].init_weights()
    def freeze_params(self):
        """Freeze all image branch parameters."""
        if self.encoders:
            for param in self.encoders["camera"]["backbone"].parameters():
                param.requires_grad = False
        # if self.with_img_backbone:
            for param in self.encoders["camera"]["neck"].parameters():
                param.requires_grad = False
        # if self.with_img_neck:
            for param in self.encoders["camera"]["vtransform"].parameters():
                param.requires_grad = False
        # if self.with_img_rpn:
            for param in self.encoders["lidar"]["voxelize"].parameters():
                param.requires_grad = False
        # if self.with_img_roi_head:
            for param in self.encoders["lidar"]["backbone"].parameters():
                param.requires_grad = False
            for param in self.decoder["backbone"].parameters():
                param.requires_grad = False
            for param in self.decoder["neck"].parameters():
                param.requires_grad = False
            for type, head in self.heads.items():
                for param in head.parameters():
                    param.requires_grad = False
    def train(self, mode=True):
        """Overload in order to keep image branch modules in eval mode."""
        super(BEVFusion, self).train(mode)
        if self.freeze:
            # if self.with_img_bbox_head:
            self.encoders["camera"]["backbone"].eval()
            # if self.with_img_backbone:
            self.encoders["camera"]["neck"].eval()
            # if self.with_img_neck:
            self.encoders["camera"]["vtransform"].eval()
            self.encoders["lidar"]["voxelize"].eval()
            self.encoders["lidar"]["backbone"].eval()
            self.decoder["backbone"].eval()
            self.decoder["neck"].eval()
            for type, head in self.heads.items():
                head.eval()
            # # if self.with_img_rpn:
            # self.img_rpn_head.eval()
            # # if self.with_img_roi_head:
            # self.img_roi_head.eval()
    def extract_camera_features(
        self,
        x,
        points,
        depth_gt,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        img_metas,
    ) -> torch.Tensor:
        B, N, C, H, W = x.size()
        x = x.view(B * N, C, H, W)

        x = self.encoders["camera"]["backbone"](x)
        x = self.encoders["camera"]["neck"](x)

        if not isinstance(x, torch.Tensor):
            x = x[0]

        BN, C, H, W = x.size()
        x = x.view(B, int(BN / B), C, H, W)
        TwoDfeature = x
        x, depth = self.encoders["camera"]["vtransform"](
            x,
            points,
            depth_gt,
            camera2ego,
            lidar2ego,
            lidar2camera,
            lidar2image,
            camera_intrinsics,
            camera2lidar,
            img_aug_matrix,
            lidar_aug_matrix,
            img_metas,
        )
        return x, depth, TwoDfeature
    def extract_camera_features_student(
        self,
        x,
        points,
        depth_gt,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        img_metas,
    ) -> torch.Tensor:
        B, N, C, H, W = x.size()
        x = x.view(B * N, C, H, W)
        # macs, params = flops_counter(self.encoders_student["camera"]["backbone"], inputs=(x,)) 
        # macs, params = clever_format([macs, params], "%.3f")
        # print("$$$$$$$$$$$$$$$$$$$$$macscamera_backbone, paramscamera_backbone: ", macs, params)
        
        x = self.encoders_student["camera"]["backbone"](x)
        # macs, params = flops_counter(self.encoders_student["camera"]["neck"], inputs=(x,))
        # macs, params = clever_format([macs, params], "%.3f")
        # print("$$$$$$$$$$$$$$$$$$$$$macscamera_neck, paramscamera_neck: ", macs, params)
        x = self.encoders_student["camera"]["neck"](x)

        if not isinstance(x, torch.Tensor):
            x = x[0]

        BN, C, H, W = x.size()
        TwoDfeature = x.view(B, int(BN / B), C, H, W)
        # macs, params = flops_counter(self.encoders_student["camera"]["vtransform"], inputs=(TwoDfeature,
        #                 points,
        #                             depth_gt,
        #                                         camera2ego,
        #                                                     lidar2ego,
        #                                                                 lidar2camera,
        #                                                                             lidar2image,
        #                                                                                         camera_intrinsics,
        #                                                                                                     camera2lidar,
        #                                                                                                                 img_aug_matrix,
        #                                                                                                                             lidar_aug_matrix,
        #                                                                                                                                         img_metas,))
        # macs, params = clever_format([macs, params], "%.3f")
        # print("$$$$$$$$$$$$$$$$$$$$$macscamera_vtransform, paramscamera_vtransform: ", macs*2, params*2)
        Lift1, depth1 = self.encoders_student["camera"]["vtransform"](
            TwoDfeature,
            points,
            depth_gt,
            camera2ego,
            lidar2ego,
            lidar2camera,
            lidar2image,
            camera_intrinsics,
            camera2lidar,
            img_aug_matrix,
            lidar_aug_matrix,
            img_metas,
        )
        Lift2, depth2 = self.encoders_student["camera"]["vtransform"](
            TwoDfeature,
            points,
            depth_gt,
            camera2ego,
            lidar2ego,
            lidar2camera,
            lidar2image,
            camera_intrinsics,
            camera2lidar,
            img_aug_matrix,
            lidar_aug_matrix,
            img_metas,
        )
        return Lift1, Lift2, depth1, depth2, TwoDfeature

    def extract_lidar_features(self, x) -> torch.Tensor:
        feats, coords, sizes = self.voxelize(x)
        batch_size = coords[-1, 0] + 1
        x = self.encoders["lidar"]["backbone"](feats, coords, batch_size, sizes=sizes)
        return x

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        feats, coords, sizes = [], [], []
        for k, res in enumerate(points):
            ret = self.encoders["lidar"]["voxelize"](res)
            if len(ret) == 3:
                # hard voxelize
                f, c, n = ret
            else:
                assert len(ret) == 2
                f, c = ret
                n = None
            feats.append(f)
            coords.append(F.pad(c, (1, 0), mode="constant", value=k))
            if n is not None:
                sizes.append(n)

        feats = torch.cat(feats, dim=0)
        coords = torch.cat(coords, dim=0)
        if len(sizes) > 0:
            sizes = torch.cat(sizes, dim=0)
            if self.voxelize_reduce:
                feats = feats.sum(dim=1, keepdim=False) / sizes.type_as(feats).view(
                    -1, 1
                )
                feats = feats.contiguous()

        return feats, coords, sizes

    def sigmoid(self, x, base=2.71828):
        return 1 / (1 + torch.pow(base, -x))

    def unified_focal_loss(self, prob_volume, depth_values, interval, depth_gt, weight, gamma, alpha):
        depth_gt_volume = depth_gt.expand_as(depth_values).float()  # (b, d, h, w)
        #interval = torch.Tensor(interval)
        #print("depth_gt_volume size",depth_gt_volume.device, depth_gt_volume.size())
        #print("depth_values size",depth_values.device, depth_values.size())
        gt_index_volume = ((depth_values <= depth_gt_volume) * ((depth_values + interval) > depth_gt_volume)).type(torch.bool)#.int()#.bool()#.long()
        #print("shape of gt_index_volume", gt_index_volume)
        gt_unity_index_volume = torch.zeros_like(prob_volume, requires_grad=False)
        #print("depth_gt_volume dtype", depth_gt_volume.dtype)
        #print("gt_index_volume dtype", gt_index_volume.dtype)
        #print("prob_volume dtype", prob_volume.dtype)
        #print("depth_values dtype", depth_values.dtype)
        #print("depth_unity_index_volume dtype", gt_unity_index_volume.dtype)
        gt_unity_index_volume[gt_index_volume] = 1.0 - (depth_gt_volume[gt_index_volume] - depth_values[gt_index_volume]) / interval
        #print("gt_unity_index_volume dtype", gt_unity_index_volume.dtype)
        gt_unity, _ = torch.max(gt_unity_index_volume, dim=1, keepdim=True)
        gt_unity = torch.where(gt_unity > 0.0, gt_unity, torch.ones_like(gt_unity))  # (b, 1, h, w)
        pos_weight = (self.sigmoid((gt_unity - prob_volume).abs() / gt_unity, base=5) - 0.5) * 4 + 1  # [1, 3]
        neg_weight = (self.sigmoid(prob_volume / gt_unity, base=5) - 0.5) * 2  # [0, 1]
        focal_weight = pos_weight.pow(gamma) * (gt_unity_index_volume > 0.0).float() + alpha * neg_weight.pow(gamma) * (
                gt_unity_index_volume <= 0.0).float()

        # mask = mask.unsqueeze(1).expand_as(depth_values).float()
        with autocast(enabled=False):
            loss = nn.BCEWithLogitsLoss(prob_volume, gt_unity_index_volume, reduction="none") * focal_weight
            #loss = F.binary_cross_entropy(prob_volume, gt_unity_index_volume, reduction="none") * focal_weight #* mask).sum() / mask.sum()
        depth_loss = loss * weight
        return depth_loss
    def get_depth_loss_uni(self, prob_volume, interval, depth_gt, weight):
        fl_gamas = [2, 1, 0]
        fl_alphas = [0.75, 0.5, 0.25]
        gamma = fl_gamas[-1]
        alpha = fl_alphas[-1]
        depth_min = 1
        depth_interval = 0.5
        ndepths = 118
        dloss_typeuni = True
        #print("prob_volume.shape:",prob_volume.shape)
        #B,N,H,W = depth_gt.size()
        #depth_gt = depth_gt.view(B*N,H,W)
        #B,N,H,W = prob_volume.size()
        #print("depth_gtZZ.shape before:",depth_gt.shape)
        depth_gt = self.get_downsampled_gt_depth(depth_gt, dloss_typeuni).unsqueeze(1)
        #print("depth_gtZZ.shape:",depth_gt.shape)
        B,N,H,W = depth_gt.size()
        depth_values = np.arange(depth_min, depth_interval * (ndepths - 0.5) + depth_min, depth_interval,
                                 dtype=np.float32)
        D = len(depth_values)
        #print("D:",D)
        depth_values = torch.from_numpy(depth_values)
        #D = depth_values.shape
        #print("D:",D)
        depth_values = depth_values.unsqueeze(0).expand(B*N,D)
        #print("###depth_values before shape BN,D: ",depth_values.size())
        #D = depth_values.shape[0]
        #print("D:",D)
        depth_values = depth_values.unsqueeze(2).expand(B*N,D,H).unsqueeze(3).expand(B*N,D,H,W).cuda()

        #print("###depth_values shape: ",depth_values.shape)
        depth_loss = self.unified_focal_loss(prob_volume, depth_values, interval, depth_gt, weight, gamma, alpha)
        return depth_loss

    def get_downsampled_gt_depth(self, gt_depths, dloss_typeuni):
        """
        Input:
            gt_depths: [B, N, H, W]
        Output:
            gt_depths: [B*N*h*w, d]
        """
        B, N, H, W = gt_depths.shape
        gt_depths = gt_depths.view(
            B * N,
            H // self.downsample_factor,
            self.downsample_factor,
            W // self.downsample_factor,
            self.downsample_factor,
            1,
        )
        gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous()
        gt_depths = gt_depths.view(
            -1, self.downsample_factor * self.downsample_factor)
        gt_depths_tmp = torch.where(gt_depths == 0.0,
                                    1e5 * torch.ones_like(gt_depths),
                                    gt_depths)
        gt_depths = torch.min(gt_depths_tmp, dim=-1).values
        gt_depths = gt_depths.view(B * N, H // self.downsample_factor,
                                   W // self.downsample_factor)
        if dloss_typeuni:
            return gt_depths
        else:
            gt_depths = (gt_depths -
                     (self.dbound[0] - self.dbound[2])) / self.dbound[2]
            gt_depths = torch.where(
            (gt_depths < self.depth_channels + 1) & (gt_depths >= 0.0),
            gt_depths, torch.zeros_like(gt_depths))
            gt_depths = F.one_hot(gt_depths.long(),
                              num_classes=self.depth_channels + 1).view(
                                  -1, self.depth_channels + 1)[:, 1:]

            return gt_depths.float()


    def get_depth_loss(self, depth_labels, depth_preds):
        dloss_typeuni = False
        depth_labels = self.get_downsampled_gt_depth(depth_labels, dloss_typeuni)
        depth_preds = depth_preds.permute(0, 2, 3, 1).contiguous().view(
            -1, self.depth_channels)
        fg_mask = torch.max(depth_labels, dim=1).values > 0.0

        with autocast(enabled=False):
            #print("depth_preds[fg_mask]",depth_preds[fg_mask].shape)
            #print("depth_labels[fg_mask]",depth_labels[fg_mask].shape)
            depth_preds = torch.where(torch.isnan(depth_preds), torch.zeros_like(depth_preds), depth_preds)
            depth_preds = torch.where(torch.isinf(depth_preds), torch.zeros_like(depth_preds), depth_preds)
            depth_loss = (F.binary_cross_entropy(
                depth_preds[fg_mask],
                depth_labels[fg_mask],
                reduction='none',
            ).sum() / max(1.0, fg_mask.sum()))

        return 3.0 * depth_loss

    def get_targets_mask(self, gt_bboxes_3d, gt_labels_3d):
        heatmaps, anno_boxes, inds, masks = multi_apply(
                CenterHead.get_targets_single(CenterHead, gt_bboxes_3d, gt_labels_3d))
        heatmaps = list(map(list, zip(*heatmaps)))
        heatmaps = [torch.stack(hms_) for hms_ in heatmaps]
        overallHeatmap = torch.zeros_like(heatmaps[0])
        for i in range(len(heatmaps)):
            overallHeatmap += heatmaps[i]
        return overallHeatmap

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    @auto_fp16(apply_to=("img", "points"))
    def forward(
        self,
        img,
        points,
        depth_gt,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        metas,
        gt_masks_bev=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        **kwargs,
    ):
        # print(f"img size: {img.shape}")
        if isinstance(img, list):
            raise NotImplementedError
        else:

            outputs = self.forward_single(
                img,
                points,
                depth_gt,
                camera2ego,
                lidar2ego,
                lidar2camera,
                lidar2image,
                camera_intrinsics,
                camera2lidar,
                img_aug_matrix,
                lidar_aug_matrix,
                metas,
                gt_masks_bev,
                gt_bboxes_3d,
                gt_labels_3d,
                **kwargs,
            )
            return outputs


    @auto_fp16(apply_to=("img", "points"))
    def forward_single(
        self,
        img,
        points,
        depth_gt,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        metas,
        gt_masks_bev=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        **kwargs,
    ):
        features = []
        for sensor in (
            self.encoders if self.training else list(self.encoders.keys())[::-1]
        ):
            if sensor == "camera":
                if self.training:
                    with torch.no_grad():
                        feature, teacher_depth, TwoDfeature_teacher = self.extract_camera_features(
                        img,
                        points,
                        depth_gt,
                        camera2ego,
                        lidar2ego,
                        lidar2camera,
                        lidar2image,
                        camera_intrinsics,
                        camera2lidar,
                        img_aug_matrix,
                        lidar_aug_matrix,
                        metas,
                    )
                    img_feature_teacher = feature
          #      print("camera feature size: ", feature.shape)
                feature_student_Lift1,  feature_student_Lift2, student_depth1, student_depth2, TwoDfeature_student = self.extract_camera_features_student(
                    img,
                    points,
                    depth_gt,
                    camera2ego,
                    lidar2ego,
                    lidar2camera,
                    lidar2image,
                    camera_intrinsics,
                    camera2lidar,
                    img_aug_matrix,
                    lidar_aug_matrix,
                    metas,
                )

            elif sensor == "lidar":
                if self.training:
                    feature = self.extract_lidar_features(points)
                    lidar_feature = feature
           #     print("lidar feature size: ", feature.shape)
            else:
                raise ValueError(f"unsupported sensor: {sensor}")
            if self.training: 
                features.append(feature)

        if not self.training:
            # avoid OOM
            features = features[::-1]
            #feature_student = feature_student[::-1]
        #print("feature_student_Lift1 shape", feature_student_Lift1.shape)
        #filename = metas[0]['filename']
        #lidar_path = metas[0]['lidar_path']
        #print(lidar_path)
        if self.bev_deformAttn:

            src = feature_student_Lift1
            # macs, params = flops_counter(self.deformAttn, inputs=(src,))
            # macs, params = clever_format([macs, params], "%.3f")
            # print("macs_DA, params_DA: ", macs, params)
            feature_student_Lift1 = self.deformAttn(src)[0] + src
        if self.bev_deform:
            src = feature_student_Lift1
            feature_student_Lift1 = self.deform(src) + src


        #save_feature_to_img_cam(feature_student_Lift2, 1, lidar_path)
        #save_feature_to_img_lidarbranch(feature_student_Lift1, 1, lidar_path)
        feature_student = [feature_student_Lift1, feature_student_Lift2]
        
        if self.fuser is not None:
            if self.training:
                xF = self.fuser(features)
            # macs, params = flops_counter(self.fuser_student, inputs=(feature_student,))
            # macs, params = clever_format([macs, params], "%.3f")
            # print("$$$$$$$$$$$$$$$$$$$$$macscamera_fuser_student, paramscamera_fuser_student: ", macs, params)
            feature_studentx = self.fuser_student(feature_student)
            # macs, params = flops_counter(self.trans_student, inputs=([feature_student_Lift1],))
            # macs, params = clever_format([macs, params], "%.3f")
            # print("$$$$$$$$$$$$$$$$$$$$$macscamera_trans_student, paramscamera_trans_student: ", macs, params)

            feature_student_plidar = self.trans_student([feature_student_Lift1])
        else:
            assert len(features) == 1, features
            x = features[0]
        x_studentF = feature_studentx
        if self.training:
            outputs = {}
            if self.fusion_feature:
                outputs['loss/fusionfeaturel2loss'] = self.feature_loss(feature_studentx, xF)
            if self.img_feature:
                outputs['loss/imgfeaturel2loss'] = self.feature_loss(feature_student_Lift2, img_feature_teacher)#*0.1
                
            if self.sem_feature:
                outputs['loss/semfeaturel2loss'] = self.feature_loss(TwoDfeature_student, TwoDfeature_teacher)
            #if self.plidar_feature:
            #    if not self.plidar_mask:
            #        outputs['loss/plidarfeaturel2loss'] = self.feature_loss(feature_student_plidar, lidar_feature)
            #    else:
                    #print("---------------------", type(gt_labels_3d))
            #        object_mask = self.get_targets_mask(gt_bboxes_3d, gt_labels_3d)
            #        outputs['loss/plidarfeaturel2loss'] = self.feature_loss(feature_student_plidar * object_mask,lidar_feature * object_mask)

            if self.depth_sup:
                depth_gt_list = depth_gt
                depth_gt = torch.stack([depth for depth in depth_gt_list]).squeeze()
                outputs['loss/depthloss'] = self.get_depth_loss(depth_gt, student_depth1)
        batch_size = x_studentF.shape[0]
        # print("after encoder-----------")
        if self.training:
            x = self.decoder["backbone"](xF)
            x = self.decoder["neck"](x)
        #print("after neck teacher x: ", x.shape)
        # macs, params = flops_counter(self.decoder_student["backbone"], inputs=(x_studentF,))
        # macs, params = clever_format([macs, params], "%.3f")
        # print("$$$$$$$$$$$$$$$$$$$$$macs_decoder_student_backbone, params_decoder_student_backbone: ", macs, params)

        x_student = self.decoder_student["backbone"](x_studentF)
        # macs, params = flops_counter(self.decoder_student["neck"], inputs=(x_student,))
        # macs, params = clever_format([macs, params], "%.3f")
        # print("$$$$$$$$$$$$$$$$$$$$$macs_decoder_student_neck, params_decoder_student_neck: ", macs, params)

        x_student = self.decoder_student["neck"](x_student)
        #print("after neck teacher x_student: ", x_student.shape)
        # print("after decoder-----------")
        if self.training:
            # outputs = {}
            # outputs['loss/featurel2loss'] = self.feature_loss(x_studentF, xF)
            for type, head in self.heads.items():
                if type == "object":
                    if self.prediction_distill:
                        pred_dict = head(x, metas)
                    else:
                        pred_dict = []
         #           print("before head: ", len(x_student), x_student[-1].shape)
                    # macs, params = flops_counter(self.heads_student[type], inputs=(x_student, metas,))
                    # macs, params = clever_format([macs, params], "%.3f")
                    # print("$$$$$$$$$$$$$$$$$$$$$macs_decoder_heads_student, params_decoder_heads_student: ", macs, params)

                    pred_dict_student = self.heads_student[type](x_student, metas)
                    # print("after head-----------")
                    #print("in head loss: ",type(gt_labels_3d))
                    losses, object_mask = head.loss(gt_bboxes_3d, gt_labels_3d, pred_dict_student, pred_dict)
                    # print("after loss-----------")
                    if self.plidar_feature:
                        if not self.plidar_mask:
                            outputs['loss/plidarfeaturel2loss'] = self.feature_loss(feature_student_plidar,
                                                                                            lidar_feature)
                        else:
                            outputs['loss/plidarfeaturel2loss'] = 100*self.feature_loss(
                                                                    feature_student_plidar * object_mask,
                                                                    lidar_feature * object_mask)
                elif type == "map":
                    losses = head(x, gt_masks_bev)
                else:
                    raise ValueError(f"unsupported head: {type}")
                for name, val in losses.items():
                    if val.requires_grad:
                        outputs[f"loss/{type}/{name}"] = val * self.loss_scale[type]
                    else:
                        outputs[f"stats/{type}/{name}"] = val
            # for type, head in self.heads_student.items():
            #     if type == "object":
            #         pred_dict = head(x_student, metas)
            #         losses = head.loss(gt_bboxes_3d, gt_labels_3d, pred_dict)
            #     elif type == "map":
            #         losses = head(x_student, gt_masks_bev)
            #     else:
            #         raise ValueError(f"unsupported head: {type}")
            #     for name, val in losses.items():
            #         if val.requires_grad:
            #             outputs[f"loss/student_{type}/{name}"] = val * self.loss_scale[type]
            #         else:
            #             outputs[f"stats/student_{type}/{name}"] = val
            # outputs["loss/kd_cls_loss"] = self.get_cls_layer_loss(teacher_cls_preds,student_cls_preds,num_class)
            # outputs["loss/kd_bbox_loss"] = self.get_box_reg_layer_loss(teacher_box_preds,student_box_preds,teacher_cls_preds,num_class)


            return outputs
        else:
            outputs = [{} for _ in range(batch_size)]
            # for type, head in self.heads.items():
            #     if type == "object":
            #         pred_dict = head(x, metas)
            #         bboxes = head.get_bboxes(pred_dict, metas)
            #         for k, (boxes, scores, labels) in enumerate(bboxes):
            #             outputs[k].update(
            #                 {
            #                     "boxes_3d": boxes.to("cpu"),
            #                     "scores_3d": scores.cpu(),
            #                     "labels_3d": labels.cpu(),
            #                 }
            #             )
            #     elif type == "map":
            #         logits = head(x)
            #         for k in range(batch_size):
            #             outputs[k].update(
            #                 {
            #                     "masks_bev": logits[k].cpu(),
            #                     "gt_masks_bev": gt_masks_bev[k].cpu(),
            #                 }
            #             )
            #     else:
            #         raise ValueError(f"unsupported head: {type}")
            for type, head in self.heads_student.items():
                if type == "object":
                    pred_dict = head(x_student, metas)
                    bboxes = head.get_bboxes(pred_dict, metas)
                    for k, (boxes, scores, labels) in enumerate(bboxes):
                        outputs[k].update(
                            {
                                "boxes_3d": boxes.to("cpu"),
                                "scores_3d": scores.cpu(),
                                "labels_3d": labels.cpu(),
                            }
                        )
                elif typee == "map":
                    logits = head(x)
                    for k in range(batch_size):
                        outputs[k].update(
                            {
                                "masks_bev": logits[k].cpu(),
                                "gt_masks_bev": gt_masks_bev[k].cpu(),
                            }
                        )
                else:
                    raise ValueError(f"unsupported head: {type}")
            return outputs

