from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from mmdeploy.core import FUNCTION_REWRITER

MAX_CLAMP_VALUE = 50000


@FUNCTION_REWRITER.register_rewriter(
    'mmdet.models.utils.vlfuse_helper.BiMultiHeadAttention.forward'
)
def bi_multi_head_attention__forward(
    self,
    vision: Tensor,
    lang: Tensor,
    attention_mask_v: Optional[Tensor] = None,
    attention_mask_l: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """Rewrite to replace masked_fill(x==0, ...) which generates Where node
    with direct arithmetic that is ONNX opset 11 compatible."""
    bsz, tgt_len, _ = vision.size()

    query_states = self.v_proj(vision) * self.scale
    key_states = self._shape(self.l_proj(lang), -1, bsz)
    value_v_states = self._shape(self.values_v_proj(vision), -1, bsz)
    value_l_states = self._shape(self.values_l_proj(lang), -1, bsz)

    proj_shape = (bsz * self.num_heads, -1, self.head_dim)
    query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    key_states = key_states.view(*proj_shape)
    value_v_states = value_v_states.view(*proj_shape)
    value_l_states = value_l_states.view(*proj_shape)

    src_len = key_states.size(1)
    attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

    if self.clamp_min_for_underflow:
        attn_weights = torch.clamp(attn_weights, min=-MAX_CLAMP_VALUE)
    if self.clamp_max_for_overflow:
        attn_weights = torch.clamp(attn_weights, max=MAX_CLAMP_VALUE)

    attn_weights_T = attn_weights.transpose(1, 2)
    attn_weights_l = (
        attn_weights_T -
        torch.max(attn_weights_T, dim=-1, keepdim=True)[0])
    if self.clamp_min_for_underflow:
        attn_weights_l = torch.clamp(attn_weights_l, min=-MAX_CLAMP_VALUE)
    if self.clamp_max_for_overflow:
        attn_weights_l = torch.clamp(attn_weights_l, max=MAX_CLAMP_VALUE)

    if attention_mask_v is not None:
        attention_mask_v = (
            attention_mask_v[:, None, None, :]
            .repeat(1, self.num_heads, 1, 1)
            .flatten(0, 1))
        attn_weights_l.masked_fill_(attention_mask_v, float('-inf'))

    attn_weights_l = attn_weights_l.softmax(dim=-1)

    if attention_mask_l is not None:
        # Original code:
        #   attention_mask = attention_mask_l.unsqueeze(1).unsqueeze(1)
        #   attention_mask = attention_mask.expand(bsz, 1, tgt_len, src_len)
        #   attention_mask = attention_mask.masked_fill(attention_mask == 0, -9e15)
        #                                               ^^^^^^^^^^^^^^^^^^^^^^^^^^^
        #                                               This generates Where node!
        #
        # Fix: use direct arithmetic instead of masked_fill(x==0, ...)
        #   (1 - mask) * -9e15  is equivalent and exports cleanly
        attention_mask = attention_mask_l.unsqueeze(1).unsqueeze(1)
        attention_mask = attention_mask.expand(bsz, 1, tgt_len, src_len)
        # Replace masked_fill(==0) with arithmetic — no Where node generated
        attention_mask = (attention_mask.float() - 1.0) * 9e15

        attn_weights = attn_weights.view(
            bsz, self.num_heads, tgt_len, src_len) + attention_mask
        attn_weights = attn_weights.view(
            bsz * self.num_heads, tgt_len, src_len)

    attn_weights_v = nn.functional.softmax(attn_weights, dim=-1)

    attn_probs_v = F.dropout(
        attn_weights_v, p=self.dropout, training=self.training)
    attn_probs_l = F.dropout(
        attn_weights_l, p=self.dropout, training=self.training)

    attn_output_v = torch.bmm(attn_probs_v, value_l_states)
    attn_output_l = torch.bmm(attn_probs_l, value_v_states)

    attn_output_v = attn_output_v.view(bsz, self.num_heads, tgt_len,
                                       self.head_dim)
    attn_output_v = attn_output_v.transpose(1, 2)
    attn_output_v = attn_output_v.reshape(bsz, tgt_len, self.embed_dim)

    attn_output_l = attn_output_l.view(bsz, self.num_heads, src_len,
                                       self.head_dim)
    attn_output_l = attn_output_l.transpose(1, 2)
    attn_output_l = attn_output_l.reshape(bsz, src_len, self.embed_dim)

    attn_output_v = self.out_v_proj(attn_output_v)
    attn_output_l = self.out_l_proj(attn_output_l)

    return attn_output_v, attn_output_l