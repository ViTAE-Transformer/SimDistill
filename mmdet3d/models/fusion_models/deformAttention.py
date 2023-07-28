import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import dtype
import torch.optim
import torch.nn as nn
from typing import Optional, List
from torch import Tensor
import math
from mmdet3d.ops.modules import MSDeformAttn
class DFA(nn.Module):
    def __init__(self, channels, bev_h, bev_w,  num_att, num_proj, nhead, npoints):
        super(DFA, self).__init__()
        self.bev_h = bev_h
        self.bev_w = bev_w
        # print(f"bev_h: {self.bev_h}, bev_w: {self.bev_w}")
        # self.uv_h = uv_h
        # self.uv_w = uv_w
        # self.M_inv = M_inv
        self.num_att = num_att
        self.num_proj = num_proj
        self.nhead = nhead
        self.npoints = npoints

        self.query_embeds = nn.ModuleList()
        self.pe = nn.ModuleList()
        self.el = nn.ModuleList()
        self.project_layers = nn.ModuleList()
        self.ref_2d = []
        self.input_spatial_shapes = []
        self.input_level_start_index = []

        bev_feat_c = channels
        for i in range(self.num_proj):
            if i > 0:
                bev_h = bev_h // 2
                bev_w = bev_w // 2
                # bev_h = uv_h // 2
                # uv_w = uv_w // 2
                if i != self.num_proj-1:
                    bev_feat_c = bev_feat_c * 2

            bev_feat_len = bev_h * bev_w
            # print(f"bev_h: {self.bev_h}, bev_w: {self.bev_w}")
            # print("bev_feat_len: ", bev_feat_len)
            #print("bev_feat_c", bev_feat_c)
            query_embed = nn.Embedding(bev_feat_len, bev_feat_c)
            self.query_embeds.append(query_embed)
            position_embed = PositionEmbeddingLearned(bev_h, bev_w, num_pos_feats=bev_feat_c//2)
            self.pe.append(position_embed)

            ref_point = self.get_reference_points(H=bev_h, W=bev_w, dim='2d', bs=1)
            self.ref_2d.append(ref_point)

            # size_top = torch.Size([bev_h, bev_w])
            # project_layer = Lane3D.RefPntsNoGradGenerator(size_top, self.M_inv, args.no_cuda)
            # self.project_layers.append(project_layer)

            spatial_shape = torch.as_tensor([(bev_h, bev_w)], dtype=torch.long)
            self.input_spatial_shapes.append(spatial_shape)

            level_start_index = torch.as_tensor([0.0,], dtype=torch.long)
            self.input_level_start_index.append(level_start_index)

            for j in range(self.num_att):
                encoder_layers = EncoderLayer(d_model=bev_feat_c, num_levels=1,
                                              num_points=self.npoints, num_heads=self.nhead)
                self.el.append(encoder_layers)

    def forward(self, bev_feature):
        projs = []
        for i in range(self.num_proj):
            if i == 0:
                bev_h = self.bev_h
                bev_w = self.bev_w
            else:
                bev_h = bev_h // 2
                bev_w = bev_w // 2
            bs, c, h, w = bev_feature.shape
            # print(" bs, c, h, w: ",  bs, c, h, w)
            # print(f"bev_h: {self.bev_h}, bev_w: {self.bev_w}")
            src = bev_feature.flatten(2).permute(0, 2, 1).float()
            #print(" src: ",  src.shape)
            # query_embed = self.query_embeds[i].weight.unsqueeze(0).repeat(bs, 1, 1)
            #query_embed = self.query_embeds[i](src.long())#.unsqueeze(0).repeat(bs, 1, 1)
            #print("query_embed shape: ", query_embed.shape)
            bev_mask = torch.zeros((bs, bev_h, bev_w), device=bev_feature.device).to(bev_feature.dtype)
            # print(f"bev_mask size: {bev_mask.shape}")
            bev_pos = self.pe[i](bev_mask).to(bev_feature.dtype)
            # print("bev_pos before flatten: ", bev_pos.shape)
            bev_pos = bev_pos.flatten(2).permute(0, 2, 1)
            #print("bev_pos after flatten: ", bev_pos.shape)
            ref_2d = self.ref_2d[i].repeat(bs, 1, 1, 1).to(bev_feature.device)
            # ref_pnts = self.project_layers[i](_M_inv).unsqueeze(-2)
            #print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&",bev_feature.device)
            input_spatial_shapes = self.input_spatial_shapes[i].to(bev_feature.device)
            input_level_start_index = self.input_level_start_index[i].to(bev_feature.device)
            for j in range(self.num_att):
                query_embed = self.el[i*self.num_att+j](query=src, value=src, bev_pos=bev_pos,
                                                        ref_2d = ref_2d,
                                                        bev_h=bev_h, bev_w=bev_w,
                                                        spatial_shapes=input_spatial_shapes,
                                                        level_start_index=input_level_start_index)
            query_embed = query_embed.permute(0, 2, 1).view(bs, c, bev_h, bev_w).contiguous()
            projs.append(query_embed)
        return projs

    @staticmethod
    def get_reference_points(H, W, Z=8, D=4, dim='3d', bs=1, device='cuda', dtype=torch.long):
        """Get the reference points used in decoder.
        Args:
            H, W spatial shape of bev
            device (obj:`device`): The device where
                reference_points should be.
        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """
        # 2d to 3d reference points, need grid from M_inv
        if dim == '3d':
            raise Exception("get reference poitns 3d not supported")
            zs = torch.linspace(0.5, Z - 0.5, D, dtype=dtype,
                                device=device).view(-1, 1, 1).expand(-1, H, W) / Z
            xs = torch.linspace(0.5, W - 0.5, W, dtype=dtype,
                                device=device).view(1, 1, W).expand(D, H, W) / W
            ys = torch.linspace(0.5, H - 0.5, H, dtype=dtype,
                                device=device).view(1, H, 1).expand(D, H, W) / H
            ref_3d = torch.stack((xs, ys, zs), -1)
            ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)

            ref_3d = ref_3d[None].repeat(bs, 1, 1, 1)
            return ref_3d
        elif dim == '2d':
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(
                    0.5, H - 0.5, H, dtype=dtype, device=device),
                torch.linspace(
                    0.5, W - 0.5, W, dtype=dtype, device=device)
            )
            ref_y = ref_y.reshape(-1)[None] / H  # ?
            ref_x = ref_x.reshape(-1)[None] / W  # ?
            ref_2d = torch.stack((ref_x, ref_y), -1)
            ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)
            return ref_2d


class EncoderLayer(nn.Module):
    '''
        one layer in Encoder,
        self-attn -> norm -> cross-attn -> norm -> ffn -> norm

        INIT:   d_model: this is C in ms uv feat map & BEV feat map
                dim_ff: num channels in feed forward net (FFN)
                activation, ffn_dropout: used in FFN
                num_levels: num layers of fpn out
                num_points, num_heads: used in deform attn
    '''

    def __init__(self,
                 d_model=None,
                 # dim_ff=None,
                 # activation="relu",
                 # ffn_dropout=0.0,
                 num_levels=1,
                 num_points=20,
                 num_heads=10):
        super().__init__()
        self.fp16_enabled = False

        self.self_attn = MSDeformAttn(d_model=d_model, n_levels=num_levels,
                                      n_heads=num_heads, n_points=num_points)  # q=v,
        # self.norm1 = nn.LayerNorm(d_model)
        #
        # self.cross_attn = DropoutMSDeformAttn(d_model=d_model,
        #                                       n_levels=num_levels,
        #                                       n_points=num_points,
        #                                       n_heads=num_heads)
        # self.norm2 = nn.LayerNorm(d_model)
        #
        # self.ffn = FFN(d_model=d_model, dim_ff=dim_ff, activation=activation,
        #                ffn_dropout=ffn_dropout)
        # self.norm3 = nn.LayerNorm(d_model)

    '''
        INPUT:  query: (B, bev_h*bev_w, C), this is BEV feat map
                value: (B, \sum_{l=0}^{L-1} H_l \cdot W_l, C), this is ms uv feat map from FPN, C is fixed for all scale
                bev_pos: BEV feat map pos embed (B, bev_h*bev_w, C)
                ref_2d: ref pnts used in self-attn, for query (B, bev_h*bev_w, 1, 2)
                ref_3d: ref pnts used in cross-attn, for ms uv feat map from FPN, this is IMPORTANT for uv-bev transform
                        (B, bev_h*bev_w, 4, 2)
                bev_h: height of bev feat map
                bev_w: widght of bev feat map
                spatial_shapes: input spatial shapes for cross-attn, this is used to split ms uv feat map
                level_start_index: input level start index for cross-attn, this is used to split ms uv feat map

            self-attn:
                input: q=v=query, ref_pnts = ref_2d (universal sampling over query space), 1-lvl
                output: query for cross-attn

            cross-attn:
                input: q=query, v=value=ms_uv_feat_map, ref_pnts = ref_3d (this is projection from bev loc to uv loc, 
                                                                            so that attention of each bev loc 
                                                                            can focus on relative uv loc)
                output: bev feat map
    '''

    def forward(self,
                query=None,
                value=None,
                bev_pos=None,
                ref_2d=None,

                bev_h=None,
                bev_w=None,
                spatial_shapes=None,
                level_start_index=None):
        # self attention
        identity = query

        temp_key = temp_value = query
        output = self.self_attn(query + bev_pos,
                               reference_points=ref_2d,
                               input_flatten=temp_value,
                               input_spatial_shapes=torch.tensor(
                                   [[bev_h, bev_w]], device=query.device),
                               input_level_start_index=torch.tensor(
                                   [0], device=query.device))
                               #identity=identity)
        # identity = query

        # # norm 1
        # query = self.norm1(query)
        #
        # # cross attention
        # query = self.cross_attn(query,
        #                         reference_points=ref_3d,
        #                         input_flatten=value,
        #                         input_spatial_shapes=spatial_shapes,
        #                         input_level_start_index=level_start_index)
        # query = query + output
        #
        # # norm 2
        # query = self.norm2(query)
        #
        # # ffn
        # query = self.ffn(query)
        #
        # # norm 3
        # query = self.norm3(query)
        # query = query + output
        return output


class DecoderLayer(nn.Module):
    '''
        one layer in Decoder,
        self-attn -> norm -> cross-attn -> norm -> ffn -> norm

        INIT:   d_model: this is C in ms uv feat map & BEV feat map
                dim_ff: num channels in feed forward net (FFN)
                activation, ffn_dropout: used in FFN
                num_points, num_heads: used in deform attn
    '''

    def __init__(self,
                 d_model=None,
                 dim_ff=None,
                 activation="relu",
                 ffn_dropout=0.0,
                 #  num_levels=4,
                 num_points=8,
                 num_heads=8):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads)
        self.norm1 = nn.LayerNorm(d_model)

        self.cross_attn = DropoutMSDeformAttn(d_model=d_model, n_levels=1,
                                              n_points=num_points, n_heads=num_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FFN(d_model=d_model, dim_ff=dim_ff, activation=activation,
                       ffn_dropout=ffn_dropout)
        self.norm3 = nn.LayerNorm(d_model)

    '''
        INPUT:  query: (B, num_query, C), this is shrinked BEV feat map
                value: (B, bev_h*bev_w, C), this is bev feat map from encoder
                query_pos: (B, num_query, C), shrinked BEV feat map pos embed
                ref_2d: (B, num_query, 1, 2) generated from query_pos
                bev_h: height of bev feat map
                bev_w: widght of bev feat map

            self-attn:
                input: q=k=v=query
                output: query for cross-attn

            cross-attn:
                input: q=query, v=value=bev_feat_map, ref_pnts = ref_2d
                output: shrinked bev feat map
    '''

    def forward(self,
                query=None,
                value=None,
                query_pos=None,
                ref_2d=None,
                bev_h=None,
                bev_w=None):
        identity = query

        temp_key = temp_value = query
        query = self.self_attn(query=query + query_pos, key=temp_key, value=temp_value)[0]
        query = query + identity

        identity = query

        query = self.norm1(query)

        query = self.cross_attn(query,
                                reference_points=ref_2d,
                                input_flatten=value,
                                input_spatial_shapes=torch.tensor(
                                    [[bev_h, bev_w]], device=query.device),
                                input_level_start_index=torch.tensor(
                                    [0], device=query.device))
        query = query + identity

        query = self.norm2(query)

        query = self.ffn(query)

        query = self.norm3(query)

        return query

class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, h=50, w=50, num_pos_feats=256):
        super().__init__()
        self.col_embed = nn.Embedding(h, num_pos_feats)
        self.row_embed = nn.Embedding(w, num_pos_feats)
        self.reset_parameters()
        self.w = w
        self.h = h

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    # supposed input is NCHW
    def forward(self, uv_feat):
        x = uv_feat
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.row_embed(i)
        y_emb = self.col_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        return pos
