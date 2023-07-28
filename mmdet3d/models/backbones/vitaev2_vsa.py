from functools import partial
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
import numpy as np
import torch.utils.checkpoint as checkpoint
from .vitaev2_vsa_modules.NormalCell import NormalCell
from .vitaev2_vsa_modules.ReductionCell import ReductionCell
from mmcv.runner import _load_checkpoint
from mmdet.utils import get_root_logger
from mmdet.models import BACKBONES

class BasicLayer(nn.Module):
    def __init__(self, img_size=224, in_chans=3, embed_dims=64, wide_pcm=False, token_dims=64, downsample_ratios=4, kernel_size=7, RC_heads=1, NC_heads=6, dilations=[1, 2, 3, 4],
                RC_op='cat', RC_tokens_type='performer', NC_tokens_type='transformer', RC_group=1, NC_group=64, NC_depth=2, dpr=0.1, mlp_ratio=4., qkv_bias=True, 
                qk_scale=None, drop=0, attn_drop=0., norm_layer=nn.LayerNorm, class_token=False, window_size=7, 
                use_checkpoint=False, cpe=False):
        super().__init__()
        self.img_size = img_size
        self.in_chans = in_chans
        self.embed_dims = embed_dims
        self.token_dims = token_dims
        self.downsample_ratios = downsample_ratios
        self.out_size = self.img_size // self.downsample_ratios
        self.RC_kernel_size = kernel_size
        self.RC_heads = RC_heads
        self.NC_heads = NC_heads
        self.dilations = dilations
        self.RC_op = RC_op
        self.RC_tokens_type = RC_tokens_type
        self.RC_group = RC_group
        self.NC_group = NC_group
        self.NC_depth = NC_depth
        self.use_checkpoint = use_checkpoint
        self.cpe = cpe
        if downsample_ratios > 1:
            self.RC = ReductionCell(img_size, in_chans, embed_dims, wide_pcm, token_dims, downsample_ratios, kernel_size,
                            RC_heads, dilations, op=RC_op, tokens_type=RC_tokens_type, group=RC_group, cpe=cpe)
        else:
            self.RC = nn.Identity()
        self.NC = nn.ModuleList([
            NormalCell(token_dims, NC_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop,
                       drop_path=dpr[i] if isinstance(dpr, list) else dpr, norm_layer=norm_layer, class_token=class_token, group=NC_group, tokens_type=NC_tokens_type,
                       img_size=img_size // downsample_ratios, window_size=window_size, cpe=cpe)
        for i in range(NC_depth)])

    def forward(self, x, size):
        h, w = size
        x, (h, w) = self.RC(x, (h, w))
        # print(h, w)
        for nc in self.NC:
            nc.H = h
            nc.W = w
            if self.use_checkpoint:
                x = checkpoint.checkpoint(nc, x)
            else:
                x = nc(x)
            # print(h, w)
        return x, (h, w)

@BACKBONES.register_module()
class ViTAEv2_VSA(nn.Module):
    def __init__(self, 
                in_chans=3, 
                img_size=224,
                embed_dims=64, 
                token_dims=64, 
                downsample_ratios=[4, 2, 2, 2], 
                kernel_size=[7, 3, 3, 3], 
                RC_heads=[1, 1, 1, 1], 
                NC_heads=4, 
                dilations=[[1, 2, 3, 4], [1, 2, 3], [1, 2], [1, 2]],
                RC_op='cat', 
                RC_tokens_type='VSA', 
                NC_tokens_type='VSA',
                RC_group=[1, 1, 1, 1], 
                NC_group=[1, 32, 64, 64], 
                NC_depth=[2, 2, 6, 2], 
                mlp_ratio=4.,
                qkv_bias=True,
                qk_scale=None,
                drop_rate=0.,
                attn_drop_rate=0.,
                drop_path_rate=0.,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                stages=4,
                window_size=7,
                wide_pcm=False,
                cpe=False,
                out_indices=(0, 1, 2, 3),
                frozen_stages=-1,
                use_checkpoint=False,
                init_cfg=None,
                load_ema=True):
        super().__init__()

        self.stages = stages
        self.load_ema = load_ema
        self.init_cfg = init_cfg
        repeatOrNot = (lambda x, y, z=list: x if isinstance(x, z) else [x for _ in range(y)])
        self.embed_dims = repeatOrNot(embed_dims, stages)
        self.tokens_dims = token_dims if isinstance(token_dims, list) else [token_dims * (2 ** i) for i in range(stages)]
        self.downsample_ratios = repeatOrNot(downsample_ratios, stages)
        self.kernel_size = repeatOrNot(kernel_size, stages)
        self.RC_heads = repeatOrNot(RC_heads, stages)
        self.NC_heads = repeatOrNot(NC_heads, stages)
        self.dilaions = repeatOrNot(dilations, stages)
        self.RC_op = repeatOrNot(RC_op, stages)
        self.RC_tokens_type = repeatOrNot(RC_tokens_type, stages)
        self.NC_tokens_type = repeatOrNot(NC_tokens_type, stages)
        self.RC_group = repeatOrNot(RC_group, stages)
        self.NC_group = repeatOrNot(NC_group, stages)
        self.NC_depth = repeatOrNot(NC_depth, stages)
        self.mlp_ratio = repeatOrNot(mlp_ratio, stages)
        self.qkv_bias = repeatOrNot(qkv_bias, stages)
        self.qk_scale = repeatOrNot(qk_scale, stages)
        self.drop = repeatOrNot(drop_rate, stages)
        self.attn_drop = repeatOrNot(attn_drop_rate, stages)
        self.norm_layer = repeatOrNot(norm_layer, stages)
        self.wide_pcm = wide_pcm
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.use_checkpoint = use_checkpoint
        self.cpe = cpe

        depth = np.sum(self.NC_depth)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        Layers = []
        for i in range(stages):
            startDpr = 0 if i==0 else self.NC_depth[i - 1]
            Layers.append(
                BasicLayer(img_size, in_chans, self.embed_dims[i], self.wide_pcm, self.tokens_dims[i], self.downsample_ratios[i],
                self.kernel_size[i], self.RC_heads[i], self.NC_heads[i], self.dilaions[i], self.RC_op[i],
                self.RC_tokens_type[i], self.NC_tokens_type[i], self.RC_group[i], self.NC_group[i], self.NC_depth[i], dpr[startDpr:self.NC_depth[i]+startDpr],
                mlp_ratio=self.mlp_ratio[i], qkv_bias=self.qkv_bias[i], qk_scale=self.qk_scale[i], drop=self.drop[i], attn_drop=self.attn_drop[i],
                norm_layer=self.norm_layer[i], window_size=window_size, use_checkpoint=use_checkpoint, cpe=cpe)
            )
            img_size = img_size // self.downsample_ratios[i]
            in_chans = self.tokens_dims[i]
        self.layers = nn.ModuleList(Layers)
        self.num_layers = len(Layers)

        self._freeze_stages()

    def _freeze_stages(self):

        if self.frozen_stages > 0:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if self.init_cfg is not None:
            self.apply(_init_weights)
            logger = get_root_logger()
            ckpt = _load_checkpoint(
                self.init_cfg.checkpoint, logger=logger, map_location='cpu')
            if self.load_ema and 'state_dict_ema' in ckpt:
                _state_dict = ckpt['state_dict_ema']
                logger.info(f'loading from state_dict_ema')
            elif 'state_dict' in ckpt:
                _state_dict = ckpt['state_dict']
            elif 'model' in ckpt:
                _state_dict = ckpt['model']
            else:
                _state_dict = ckpt
            state_dict = OrderedDict()
            for k, v in _state_dict.items():
                if k.startswith('backbone.'):
                    state_dict[k[9:]] = v
            
            # strip prefix of state_dict
            if list(state_dict.keys())[0].startswith('module.'):
                state_dict = {k[7:]: v for k, v in state_dict.items()}


            # reshape absolute position embedding
            if state_dict.get('absolute_pos_embed') is not None:
                absolute_pos_embed = state_dict['absolute_pos_embed']
                N1, L, C1 = absolute_pos_embed.size()
                N2, C2, H, W = self.absolute_pos_embed.size()
                if N1 != N2 or C1 != C2 or L != H * W:
                    logger.warning('Error in loading absolute_pos_embed, pass')
                else:
                    state_dict['absolute_pos_embed'] = absolute_pos_embed.view(
                        N2, H, W, C2).permute(0, 3, 1, 2).contiguous()

            # interpolate position bias table if needed
            relative_position_bias_table_keys = [
                k for k in state_dict.keys()
                if 'relative_position_bias_table' in k
            ]
            for table_key in relative_position_bias_table_keys:
                table_pretrained = state_dict[table_key]
                table_current = self.state_dict()[table_key]
                L1, nH1 = table_pretrained.size()
                L2, nH2 = table_current.size()
                if nH1 != nH2:
                    logger.warning(f'Error in loading {table_key}, pass')
                elif L1 != L2:
                    S1 = int(L1**0.5)
                    S2 = int(L2**0.5)
                    table_pretrained_resized = F.interpolate(
                        table_pretrained.permute(1, 0).reshape(1, nH1, S1, S1),
                        size=(S2, S2),
                        mode='bicubic')
                    state_dict[table_key] = table_pretrained_resized.view(
                        nH2, L2).permute(1, 0).contiguous()

            # load state_dict
            msg = self.load_state_dict(state_dict, False)
            logger.info(msg)
        elif self.init_cfg is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def forwardTwoLayer(self, x):
        x1 = self.layers[0](x)
        x2 = self.layers[1](x1)
        return x1, x2
    
    def forwardThreeLayer(self, x):
        x0 = self.layers[1](x)
        x1 = self.layers[2](x0)
        x2 = self.layers[3](x1)
        return x0, x1, x2

    def forward_features(self, x):
        b, c, h, w = x.shape
        for layer in self.layers:
            x, (h, w) = layer(x, (h, w))
        return x

    def forward(self, x):
        """Forward function."""
        outs = []
        b, _, h, w = x.shape
        for layer in self.layers:
            x, (h, w) = layer(x, (h, w))
            outs.append(x.reshape(b, h, w, -1).permute(0, 3, 1, 2).contiguous())

        return tuple(outs)

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(ViTAEv2_VSA, self).train(mode)
        self._freeze_stages()
