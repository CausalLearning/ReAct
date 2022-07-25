import copy
from itertools import repeat
from typing import Optional

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.init import constant_, xavier_uniform_

from React.model.grid_sample1d import GridSample1d
from React.utill.misc import inverse_sigmoid
from React.utill.temporal_box_producess import ml2se, segment_iou


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = torch.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class Transformer(nn.Module):

    def __init__(self, num_class, d_model=512, nhead=8, encoder_sample_num=3, decoder_sample_num=4, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1, activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        self.return_intermediate_dec = return_intermediate_dec

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before, encoder_sample_num)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before, decoder_sample_num)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self.d_model = d_model
        self.nhead = nhead
        self.num_class = num_class

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, DeformableAttention):
                m._reset_parameters()

        xavier_uniform_(self.decoder.reference_point.weight.data, gain=1.0)
        constant_(self.decoder.reference_point.bias.data, 0.)

    @staticmethod
    def generate_square_subsequent_mask(sz, device):
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float(
            '-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    @staticmethod
    def get_valid_ratio(mask):
        _, T = mask.shape
        valid_T = torch.sum(~mask, 1)
        valid_ratio = valid_T.float() / T
        return valid_ratio  # shape=(bs)

    def forward(self, srcs, masks, query, pos_embeds=None, query_mask=None, gt_bbox=None, snippet_num=None):
        srcs = srcs.transpose(0, 1)  # L_max, bz, dim
        valid_ratios = self.get_valid_ratio(masks).unsqueeze(-1)  # bs, 1

        if pos_embeds is not None:
            pos_embeds = pos_embeds.permute(2, 0, 1)
        srcs = self.encoder(src=srcs, pos=pos_embeds, src_key_padding_mask=masks, valid_ratio=valid_ratios, snippet_num=snippet_num)

        hs, init_refpoint, ref_point, tgt = self.decoder(query, srcs, tgt_key_padding_mask=query_mask, valid_ratio=valid_ratios,
                                                         memory_key_padding_mask=masks,
                                                         snippet_num=snippet_num)
        if self.return_intermediate_dec:
            hs = hs.transpose(1, 2)
        else:
            hs = hs.transpose(0, 1)

        return hs, init_refpoint, ref_point, srcs.transpose(0, 1), tgt


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    @staticmethod
    def get_reference_points(temporal_dim_len, valid_ratios, device):
        # temporal_dim_len： the shape of temporal dimension after padding
        ref = torch.linspace(0.5, temporal_dim_len - 0.5, temporal_dim_len, dtype=torch.float32, device=device)
        ref = ref[None] / temporal_dim_len  # (1, temporal_dim_len)
        return ref  # (1, temporal_dim_len)

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,

                valid_ratio=None,
                snippet_num=None
                ):
        output = src

        reference_points = self.get_reference_points(src.shape[0], valid_ratio, device=src.device).expand(src.shape[1], -1).unsqueeze(-1)
        for layer in self.layers:
            output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask, pos=pos,
                           reference_point=reference_points, valid_ratio=valid_ratio, snippet_num=snippet_num)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

        self.reference_point = nn.Linear(self.layers[0].d_model, 1)

        # for segment refinement
        self.segment_embed = None

        self.return_intermediate = return_intermediate

    def feed_gt(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                gt_segments=None,
                idx=None,
                valid_ratio=None,
                snippet_num=None):
        bz = memory.shape[1]

        Lq = tgt.shape[0]
        with torch.no_grad():
            refer_seg = gt_segments.float().contiguous().detach()
            reference_point = torch.ones((bz, Lq, 2), device=tgt.device) * 0.5  # set other not-trained queries to [0.5, 0.5]
            reference_point[idx] = refer_seg
            reference_point = reference_point * valid_ratio.unsqueeze(-1)
        output = tgt
        intermediate = []

        for i, layer in enumerate(self.layers):
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           query_pos=query_pos,
                           reference_point=reference_point,
                           valid_ratio=valid_ratio,
                           snippet_num=snippet_num
                           )

            if self.return_intermediate:
                intermediate.append((output.transpose(0, 1))[idx])

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append((output.transpose(0, 1))[idx])

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                valid_ratio=None,
                snippet_num=None
                ):
        bz = memory.shape[1]

        output, tgt = torch.chunk(tgt, 2, dim=1)
        tgt = tgt.unsqueeze(0).expand(bz, -1, -1)
        output = output.unsqueeze(0).expand(bz, -1, -1).transpose(0, 1)
        query_features = output
        intermediate = []
        intermediate_reference_points = []

        reference_point = torch.sigmoid(self.reference_point(tgt))  # bz, Lq, 1
        init_reference_point = reference_point

        for i, layer in enumerate(self.layers):
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=tgt, query_pos=query_pos,
                           reference_point=reference_point * valid_ratio.unsqueeze(-1),
                           valid_ratio=valid_ratio,
                           snippet_num=snippet_num
                           )

            # segment refinement
            # update the reference point/segment of the next layer according to the output from the current layer
            tmp = self.segment_embed[i](output).transpose(0, 1)  # bz, Lq, 2
            if reference_point.shape[-1] == 1:
                new_reference_points = tmp
                new_reference_points[..., :1] = tmp[..., :1] + inverse_sigmoid(reference_point)
                new_reference_points = new_reference_points.sigmoid()
                reference_point = new_reference_points.detach()
            else:
                new_reference_points = tmp + inverse_sigmoid(reference_point)
                new_reference_points = new_reference_points.sigmoid()
                reference_point = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_point)

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate), init_reference_point, torch.stack(intermediate_reference_points), query_features

        return output, reference_point


class DeformableAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, sampling_point=4):
        super(DeformableAttention, self).__init__()

        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        self.num_heads = num_heads
        self.K = sampling_point

        # padding mode: if offset >1 or <-1, pad with boarder value, else 0.
        self.time_feature_sample = GridSample1d(padding_mode=False, align_corners=True)
        self.sampling_offsets = nn.Linear(self.embed_dim, num_heads * sampling_point)
        self.atten_weight = nn.Linear(self.embed_dim, num_heads * sampling_point)
        self.value_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.)
        # Initial offsets:
        # (1, 0, -1, 0, -1, 0, 1, 0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (4.0 * math.pi / self.num_heads)
        grid_init = thetas.cos()

        grid_init = grid_init.view(self.num_heads, 1).repeat(1, self.K)
        for i in range(self.K):
            grid_init[:, i] *= i + 1

        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.atten_weight.weight.data, 0.)
        constant_(self.atten_weight.bias.data, 0.)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.out_proj.weight.data)
        constant_(self.out_proj.bias.data, 0.)

    def forward(self, query, value, value_key_padding_mask, value_valid_ratio, reference_point, snippet_num):
        # query Lq, batch_size, Dim
        # value Lv, batch_size, Dim
        # value_valid_ratio: batch_size, 1 (no-padding feature num)/(max feautre num)

        Lq, bz, _ = query.shape
        Lv, bz, _ = value.shape

        # in_proj
        value = self.value_proj(value)

        if value_key_padding_mask is not None:
            value = value.masked_fill(value_key_padding_mask.transpose(0, 1).unsqueeze(-1), float(0))

        # prepare for value
        value = value.permute(1, 2, 0)  # bz, dim, Lv

        value = value.reshape(bz * self.num_heads, self.head_dim, Lv)  # bz*N_head, head_dim, Lv

        # sampling offset
        offset = self.sampling_offsets(query.flatten(0, 1)).view(Lq, bz, self.num_heads, self.K).permute(1, 2, 0, 3)  # bz, N_head, Lq, K

        if reference_point.shape[-1] == 2:
            # offset: bz, N_head, Lq, K , reference_point bz,Lq
            offset = reference_point[..., :1].view(bz, 1, Lq, 1) + offset / self.K * reference_point[..., 1:].view(bz, 1, Lq, 1) * 0.5
        elif reference_point.shape[-1] == 1:
            offset = reference_point.view(bz, 1, Lq, 1) + offset / snippet_num.view(-1, 1, 1, 1)

        offset = offset * 2 - 1
        offset = offset.reshape(bz * self.num_heads, Lq * self.K)

        # calculate attention weight
        attn_wight = self.atten_weight(query.flatten(0, 1)).view(Lq, bz, self.num_heads, self.K).permute(1, 2, 0, 3)  # bz, N_head, Lq, K
        attn_wight = torch.softmax(attn_wight, dim=-1)
        attn_wight = attn_wight.reshape(bz * self.num_heads * Lq, self.K, 1)  # bz*N_head*Lq, K, 1

        # sampling time fuature from value
        # sample (bz*N_head, head_dim, Lv) with offset (bz*N_head, Lq*K) ---> (bz*N_head, head_dim, Lq * K)
        value = self.time_feature_sample(value.contiguous(), offset).view(bz * self.num_heads, self.head_dim, Lq, self.K)
        value = value.transpose(1, 2).reshape(bz * self.num_heads * Lq, self.head_dim, self.K)  # bz*N_head*Lq, head_dim, K

        # calculate attention output
        attn_out = torch.matmul(value, attn_wight)  # bz*N_head*Lq, head_dim, 1

        attn_out = attn_out.view(bz, self.num_heads, Lq, self.head_dim).permute(2, 0, 1, 3)
        attn_out = attn_out.reshape(Lq, bz, self.embed_dim)  # done! Lq, batch_size, Dim

        # output proj
        attn_out = self.out_proj(attn_out)

        return attn_out


class RelationAttention(nn.Module):

    def __init__(self, embed_dim, nhead, nlayer=2):
        super(RelationAttention, self).__init__()

        self.Q = nn.Linear(embed_dim, embed_dim)
        self.K = nn.Linear(embed_dim, embed_dim)
        self.V = nn.Linear(embed_dim, embed_dim)
        self.nhead = nhead
        self.nlayer = nlayer

    def construct_graph(self, segments, query):
        with torch.no_grad():
            bz, Lq = segments.shape[0], segments.shape[1]
            adj_matrix = torch.diag(torch.ones(Lq, device=segments.device, dtype=torch.bool)).unsqueeze(0).repeat(bz, 1, 1)  # bz, Lq, Lq

            segments = ml2se(segments)
            iou = segment_iou(segments, segments)  # bz,Lq,Lq
            adj_matrix[iou <= 0.2] = 1

            cos_sim = torch.cosine_similarity(query.unsqueeze(2), query.unsqueeze(1), dim=-1)
            adj_matrix[cos_sim <= 0.2] = 0

            adj_matrix = ~adj_matrix

        return adj_matrix

    def forward(self, query, reference):
        Lq, bz, _ = query.shape
        query = query.transpose(0, 1)

        if reference.shape[-1] == 2:
            adj_matrix = self.construct_graph(reference, query)
            adj_matrix = adj_matrix.unsqueeze(1).expand(-1, self.nhead, -1, -1).flatten(0, 1)

        h = query
        Q = self.Q(h).view(bz, Lq, self.nhead, -1).permute(0, 2, 1, 3).reshape(bz * self.nhead, Lq, -1)  # bz*n_head, Lq, head_dim
        K = self.K(h).view(bz, Lq, self.nhead, -1).permute(0, 2, 1, 3).reshape(bz * self.nhead, Lq, -1)
        V = self.V(h).view(bz, Lq, self.nhead, -1).permute(0, 2, 1, 3).reshape(bz * self.nhead, Lq, -1)
        attn = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(Q.shape[-1])

        if reference.shape[-1] == 2:
            attn = attn.masked_fill(adj_matrix, float("-inf"))

        attn = torch.softmax(attn, dim=-1)

        h_res = torch.matmul(attn, V)
        h = h_res.view(bz, self.nhead, Lq, -1).permute(0, 2, 1, 3).reshape(bz, Lq, -1) + h
        h = h.transpose(0, 1)  # Lq, bz, dim
        return h


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, sample_num=3):
        super().__init__()

        self.self_attn = DeformableAttention(d_model, num_heads=nhead, sampling_point=sample_num)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     reference_point=None,
                     valid_ratio=None,
                     snippet_num=None
                     ):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(query=q, value=src, value_key_padding_mask=src_key_padding_mask, value_valid_ratio=valid_ratio,
                              reference_point=reference_point, snippet_num=snippet_num)

        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,

                    reference_point=None,
                    valid_ratio=None,
                    snippet_num=None
                    ):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)

        src2 = self.self_attn(query=q, value=src, value_key_padding_mask=src_key_padding_mask, value_valid_ratio=valid_ratio,
                              reference_point=reference_point, snippet_num=snippet_num)
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,

                reference_point=None,
                valid_ratio=None,
                snippet_num=None
                ):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos,
                                    reference_point=reference_point, valid_ratio=valid_ratio, snippet_num=snippet_num
                                    )
        return self.forward_post(src, src_mask, src_key_padding_mask, pos,
                                 reference_point=reference_point, valid_ratio=valid_ratio, snippet_num=snippet_num
                                 )


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, sample_num=3):
        super().__init__()
        self.self_attn = RelationAttention(d_model, nhead, nlayer=1)
        self.cross_attn = DeformableAttention(d_model, num_heads=nhead, sampling_point=sample_num)

        self.d_model = d_model
        self.num_heads = nhead
        self.K = sample_num
        self.head_dim = d_model // nhead
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                reference_point=None,
                valid_ratio=None,
                snippet_num=None
                ):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, reference_point)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.cross_attn(query=tgt2, value=memory, value_key_padding_mask=memory_key_padding_mask, value_valid_ratio=valid_ratio,
                               reference_point=reference_point, snippet_num=snippet_num)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
