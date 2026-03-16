from typing import Dict, List, Optional, Tuple, Union

import math
import torch
from torch import nn
from torch import Tensor

from mmdet.models.dense_heads import GroundingDINOHead
from mmdet.models.losses import QualityFocalLoss
from mmdet.registry import MODELS
from mmdet.utils import InstanceList, reduce_mean

from mmengine.structures import InstanceData
from mmdet.structures import SampleList
from mmdet.structures.bbox import bbox_cxcywh_to_xyxy
from mmdet.models.dense_heads.atss_vlfusion_head import convert_grounding_to_cls_scores
from mmdet.models.utils import multi_apply

from .decoder import GroundingPOSETransformerDecoder



class ContrastiveEmbed(nn.Module):
    """text visual ContrastiveEmbed layer.

    Args:
        log_scale (Optional[Union[str, float]]):  The initial value of a
          learnable parameter to multiply with the similarity
          matrix to normalize the output.  Defaults to 0.0.
          - If set to 'auto', the similarity matrix will be normalized by
            a fixed value ``sqrt(d_c)`` where ``d_c`` is the channel number.
          - If set to 'none' or ``None``, there is no normalization applied.
          - If set to a float number, the similarity matrix will be multiplied
            by ``exp(log_scale)``, where ``log_scale`` is learnable.
        bias (bool, optional): Whether to add bias to the output.
          If set to ``True``, a learnable bias that is initialized as -4.6
          will be added to the output. Useful when training from scratch.
          Defaults to False.
    """

    def __init__(self,
                 log_scale: Optional[Union[str, float]] = None,
                 bias: bool = False):
        super().__init__()
        self.log_scale = log_scale
        if isinstance(log_scale, float):
            self.log_scale = nn.Parameter(
                torch.Tensor([float(log_scale)]), requires_grad=True)
        elif log_scale not in ['auto', 'none', None]:
            raise ValueError(f'log_scale should be one of '
                             f'"auto", "none", None, but got {log_scale}')

        self.bias = None
        if bias:
            bias_value = -math.log((1 - 0.01) / 0.01)
            self.bias = nn.Parameter(
                torch.Tensor([bias_value]), requires_grad=True)

    def forward(self, visual_feat: Tensor, text_feat: Tensor,
                text_token_mask: Tensor) -> Tensor:
        """Forward function.

        Args:
            visual_feat (Tensor): Visual features, has shape (batch_size, num_queries, embed_dims)
            text_feat (Tensor): Text features, has shape (batch_size, num_text_tokens, embed_dims)
            text_token_mask (Tensor): A mask used for text feats, has shape (batch_size, num_text_tokens)

        Returns:
            Tensor: Classification score, has shape (batch_size, num_queries, num_text_tokens)
        """
        res = visual_feat @ text_feat.transpose(-1, -2)
        if isinstance(self.log_scale, nn.Parameter):
            res = res * self.log_scale.exp()
        elif self.log_scale == 'auto':
            # NOTE: similar to the normalizer in self-attention
            res = res / math.sqrt(visual_feat.shape[-1])
        if self.bias is not None:
            res = res + self.bias
        res = res.masked_fill(~text_token_mask[:, None, :], float('-inf'))

        return res


class RelationBranch(nn.Module):
    """Relation branch for Grounding DINO with cross-attention to image (memory).
    Uses text prompts to predict relationships between keypoints.

    Args:
        embed_dims_in (int): Dimension of the input node embeddings.
        embed_dims (int): Dimension of the embedding.
        relation_decoder (dict): Config for the decoder with cross-attention.
        dropout (float): Dropout rate.
    """

    def __init__(
        self, 
        embed_dims_in: int,
        embed_dims: int, 
        relation_decoder: dict,
    ):
        super().__init__()
        
        self.node_pair_to_relation_feat = nn.Linear(embed_dims_in * 2, embed_dims)

        self.memory_proj = nn.Linear(embed_dims_in, embed_dims)
        self.memory_relation_proj = nn.Linear(embed_dims_in, embed_dims)

        assert 'self_attn_cfg' not in relation_decoder['layer_cfg']
        relation_decoder['layer_cfg']['self_attn_cfg'] = dict(ignore_self_attn=True)
        self.decoder = GroundingPOSETransformerDecoder(**relation_decoder)

        self.relation_classifier = ContrastiveEmbed(log_scale=0.0)

    def build_relation_features(self, node_features: Tensor) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """Generate pairwise relation features from node features.
        
        Args:
            node_features (Tensor): Node embeddings of shape (num_nodes, embed_dims)
            
        Returns:
            Tuple[Tensor, Tuple[Tensor, Tensor]]: 
                - Relation features of shape (num_pairs, embed_dims)
                - Tuple of (row_indices, col_indices) for the pairs
        """
        n_nodes = node_features.shape[0]
        
        # Get indices for upper triangular part (excluding diagonal)
        row_indices, col_indices = torch.triu_indices(n_nodes, n_nodes, offset=1)
        row_indices = row_indices.to(node_features.device)
        col_indices = col_indices.to(node_features.device)
        
        # Get features for selected pairs only
        q1 = node_features[row_indices]  # (num_pairs, embed_dims)
        q2 = node_features[col_indices]  # (num_pairs, embed_dims)
        
        # Concatenate and process through MLP
        pair_features = torch.cat([q1, q2], dim=-1)  # (num_pairs, embed_dims*2)
        relation_features = self.node_pair_to_relation_feat(pair_features)  # (num_pairs, embed_dims)
        
        return relation_features, (row_indices, col_indices)

    def forward_single_train(self, 
                        hidden_states: Tensor,
                        memory_item: Tensor,
                        memory_mask_item: Tensor,
                        references_item: Tensor,
                        valid_ratios_item: Tensor,
                        spatial_shapes: Tensor,
                        level_start_index: Tensor,
                        memory_relation_text_item: Tensor,
                        relation_text_token_mask_item: Tensor) -> Tensor:
        """Process a single batch item to generate relation predictions.

        Args:
            hidden_states (Tensor): Hidden states of shape (num_valid, embed_dims).
            memory_item (Tensor): Memory item of shape (1, num_memory, embed_dims).
            memory_mask_item (Tensor): Memory mask item of shape (1, num_memory).
            references_item (Tensor): Reference points of shape (num_valid, 4).
            valid_ratios_item (Tensor): Valid ratios of shape (num_valid, 2).
            spatial_shapes (Tensor): Spatial shapes of shape (num_valid, 2).
            level_start_index (Tensor): Level start index of shape (num_valid,).
            memory_relation_text_item (Tensor): Memory relation text item of shape (1, num_memory_text, embed_dims).
            relation_text_token_mask_item (Tensor): Relation text token mask item of shape (1, num_memory_text).

        Returns:
            Tensor: Relation scores matrix for valid nodes,
                shape (num_valid, num_valid, len_text) if return_intermediate is False,
                otherwise shape (num_decoder_layer, num_valid, num_valid, len_text)
        """
        n_valid = hidden_states.shape[0]
        
        relation_features, (row_indices, col_indices) = self.build_relation_features(hidden_states)
        num_pairs = relation_features.shape[0]
        
        src_refs = references_item[row_indices, :2]
        dst_refs = references_item[col_indices, :2]
        relation_reference_points = (src_refs + dst_refs) / 2
        relation_reference_points = relation_reference_points.unsqueeze(0)

        refined_relation_features, _ = self.decoder(
            query=relation_features.unsqueeze(0),
            value=memory_item.unsqueeze(0),
            reference_points=relation_reference_points,
            valid_ratios=valid_ratios_item.unsqueeze(0),
            key_padding_mask=memory_mask_item.unsqueeze(0) if memory_mask_item is not None else None,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            memory_text=memory_relation_text_item.unsqueeze(0),
            text_attention_mask=~relation_text_token_mask_item.unsqueeze(0),
            reg_branches=None,
            self_attn_mask=None
        )
        
        num_layers = refined_relation_features.shape[0]
        refined_relation_features = refined_relation_features.reshape(num_layers, num_pairs, -1)
        
        relation_scores = torch.zeros(
            (num_layers, n_valid, n_valid, memory_relation_text_item.shape[0]),
            dtype=torch.float32, device=hidden_states.device
        )
        
        for layer_idx in range(num_layers):
            layer_features = refined_relation_features[layer_idx]
            layer_scores = self.relation_classifier(
                layer_features,
                memory_relation_text_item.unsqueeze(0),
                relation_text_token_mask_item.unsqueeze(0)
            )
            
            # Fill upper triangular part
            relation_scores[layer_idx, row_indices, col_indices] = layer_scores
            # Fill lower triangular part (symmetric matrix)
            relation_scores[layer_idx, col_indices, row_indices] = layer_scores

        return relation_scores

    

    def forward(self, 
                hidden_states: Tensor,
                memory: Tensor,
                memory_mask: Tensor,
                spatial_shapes: Tensor,
                level_start_index: Tensor,
                valid_ratios: Tensor,
                references: Tensor,
                memory_relation_text: Tensor,
                relation_text_token_mask: Tensor,
                mask: Optional[Tensor] = None) -> Tensor:
        """
        
        Args:
            hidden_states: Hidden states with shape (batch_size, num_queries, embed_dims)
            ...
            mask: Boolean mask with shape (batch_size, num_queries)

        Returns:
            Tensor: Relation predictions with shape (batch_size, num_decoder_layers, num_queries, num_queries, len_text)
        """
        batch_size, num_queries, embed_dims = hidden_states.shape

        memory = self.memory_proj(memory)
        memory_relation_text = self.memory_relation_proj(memory_relation_text)
        
        batch_relations = torch.zeros(
            (batch_size, self.decoder.num_layers if self.decoder is not None else 1, num_queries, num_queries, memory_relation_text.shape[1]),
            device=hidden_states.device
        )
        
        for b_idx in range(0, batch_size):
            valid_indices = torch.where(mask[b_idx])[0] if mask is not None else torch.arange(num_queries, device=hidden_states.device)
            
            if len(valid_indices) <= 1:  # Skip if 0 or 1 valid nodes (no pairs)
                continue
            
            relations = self.forward_single_train(
                hidden_states=hidden_states[b_idx, valid_indices],
                memory_item=memory[b_idx],
                memory_mask_item=memory_mask[b_idx] if memory_mask is not None else None,
                references_item=references[b_idx, valid_indices] if references is not None else None,
                valid_ratios_item=valid_ratios[b_idx],
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                memory_relation_text_item=memory_relation_text[b_idx],
                relation_text_token_mask_item=relation_text_token_mask[b_idx],
            )
            
            batch_relations[b_idx, :, valid_indices[:, None], valid_indices] = relations

        return batch_relations

    def predict(
        self,
        hidden_states: Tensor,
        memory: Tensor,
        memory_mask: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        valid_ratios: Tensor,
        references: Tensor,
        memory_relation_text: Tensor,
        relation_text_token_mask: Tensor,
        batch_data_samples: SampleList
    ):

        B, N, D = hidden_states.shape

        memory = self.memory_proj(memory)
        memory_relation_text = self.memory_relation_proj(memory_relation_text)

        # ---- build pair indices once ----
        row_idx, col_idx = torch.triu_indices(N, N, offset=1, device=hidden_states.device)

        # ---- gather node features ----
        q1 = hidden_states[:, row_idx]      # (B, num_pairs, D)
        q2 = hidden_states[:, col_idx]      # (B, num_pairs, D)

        pair_features = torch.cat([q1, q2], dim=-1)
        relation_features = self.node_pair_to_relation_feat(pair_features)  # (B, num_pairs, D)

        # ---- reference points ----
        src_refs = references[:, row_idx, :2]
        dst_refs = references[:, col_idx, :2]
        relation_reference_points = (src_refs + dst_refs) / 2  # (B, num_pairs, 2)

        # ---- decoder ----
        refined_relation_features, _ = self.decoder(
            query=relation_features,
            value=memory,
            reference_points=relation_reference_points,
            valid_ratios=valid_ratios,
            key_padding_mask=memory_mask,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            memory_text=memory_relation_text,
            text_attention_mask=~relation_text_token_mask,
            reg_branches=None,
            self_attn_mask=None
        )

        refined_relation_features = refined_relation_features[-1]  # (B, num_pairs, D)

        # ---- classification ----
        scores = self.relation_classifier(
            refined_relation_features,
            memory_relation_text,
            relation_text_token_mask
        )  # (B, num_pairs, num_text)

        # ---- scatter to full matrix ----
        relation_scores = torch.zeros(
            B, N, N, scores.shape[-1],
            device=hidden_states.device
        )

        relation_scores[:, row_idx, col_idx] = scores
        relation_scores[:, col_idx, row_idx] = scores

        # ---- convert grounding logits ----
        relation_predictions = []

        for i in range(B):
            rel = relation_scores[i]

            rel = rel.reshape(N * N, -1)

            rel = convert_grounding_to_cls_scores(
                logits=rel.sigmoid()[None],
                positive_maps=[batch_data_samples[i].token_positive_map_relation]
            )[0]

            rel = rel.reshape(N, N, -1)

            relation_predictions.append(rel)

        return relation_predictions


@MODELS.register_module()
class GroundingPOSEHead(GroundingDINOHead):

    def __init__(self, 
                 relation_branch: dict,
                 loss_relation: dict,
                 **kwargs):
        self.relation_branch_cfg = relation_branch
        self.loss_relation_cfg = loss_relation

        super().__init__(**kwargs)

    def _init_layers(self) -> None:
        super()._init_layers()
        self.loss_relation = MODELS.build(self.loss_relation_cfg)

        self.relation_branch = RelationBranch(**self.relation_branch_cfg)


    def predict(self,
                hidden_states: Tensor,
                references: List[Tensor],
                memory_text: Tensor,
                text_token_mask: Tensor,
                memory_relation_text: Tensor,
                relation_text_token_mask: Tensor,
                batch_data_samples: SampleList,
                memory: Tensor,
                memory_mask: Tensor,
                spatial_shapes: Tensor,
                level_start_index: Tensor,
                valid_ratios: Tensor,
                rescale: bool = True) -> InstanceList:
        predictions, relation_predictions = self.predict_onnx(hidden_states, references, memory_text, text_token_mask, memory_relation_text, relation_text_token_mask, batch_data_samples, memory, memory_mask, spatial_shapes, level_start_index, valid_ratios, rescale)
        
        results = []
        for (scores, labels, keypoints, _), relation_scores in zip(predictions, relation_predictions):
            instance = InstanceData()
            instance.scores = scores
            instance.labels = labels
            instance.keypoints = keypoints
            instance.relation_scores = relation_scores
            results.append(instance)
   
        return results

    def predict_onnx(self,
                hidden_states: Tensor,
                references: List[Tensor],
                memory_text: Tensor,
                text_token_mask: Tensor,
                memory_relation_text: Tensor,
                relation_text_token_mask: Tensor,
                batch_data_samples: SampleList,
                memory: Tensor,
                memory_mask: Tensor,
                spatial_shapes: Tensor,
                level_start_index: Tensor,
                valid_ratios: Tensor,
                rescale: bool = True) -> InstanceList:
        
        predictions = super().predict(
            hidden_states=hidden_states,
            references=references,
            memory_text=memory_text,
            text_token_mask=text_token_mask,
            batch_data_samples=batch_data_samples,
            rescale=rescale)

        hidden_states_indexed = torch.stack([
            hidden_states[-1, i, indexes] for i, (_, _, _, indexes) in enumerate(predictions)
        ])
        references_indexed = torch.stack([
            references[-1][i, indexes] for i, (_, _, _, indexes) in enumerate(predictions)
        ])

        relation_predictions = self.relation_branch.predict(
            hidden_states_indexed,
            memory=memory,
            memory_mask=memory_mask,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            references=references_indexed,
            memory_relation_text=memory_relation_text,
            relation_text_token_mask=relation_text_token_mask,
            batch_data_samples=batch_data_samples
        )

        return predictions, relation_predictions
    
    def _predict_by_feat_single(self,
                                cls_score: Tensor,
                                bbox_pred: Tensor,
                                token_positive_maps: dict,
                                img_meta: dict,
                                rescale: bool = True) -> InstanceData:
        """Transform a single image's features extracted from the head into
        bbox results.

        Args:
            cls_score (Tensor): Box score logits from the last decoder layer
                for each image. Shape [num_queries, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from the last decoder layer
                for each image, with coordinate format (cx, cy) and
                shape [num_queries, 2].
            token_positive_maps (dict): Token positive map.
            img_meta (dict): Image meta info.
            rescale (bool, optional): If True, return boxes in original image
                space. Default True.

        Returns:
            - scores (Tensor): Classification scores, has a shape
                (num_instance, )
            - labels (Tensor): Labels of bboxes, has a shape
                (num_instances, ).
            - keypoints (Tensor): The center of the bbox, has a shape
                (num_instances, 2), the last dimension 2 arrange as (cx, cy).
            - indexes (Tensor): The index of the bbox in the original
                `bbox_pred` tensor, has a shape (num_instances, ).
        """
        assert len(cls_score) == len(bbox_pred)  # num_queries
        max_per_img = self.test_cfg.get('max_per_img', len(cls_score))
        img_shape = img_meta['img_shape']

        cls_score = convert_grounding_to_cls_scores(
            logits=cls_score.sigmoid()[None],
            positive_maps=[token_positive_maps])[0]
        scores, indexes = cls_score.view(-1).topk(max_per_img)
        num_classes = cls_score.shape[-1]
        det_labels = indexes % num_classes
        bbox_index = indexes // num_classes
        bbox_pred = bbox_pred[bbox_index]

        det_keypoints = bbox_pred[:, :2]
        det_keypoints[:, 0] *= img_shape[1]
        det_keypoints[:, 1] *= img_shape[0]
        det_keypoints[:, 0].clamp_(min=0, max=img_shape[1])
        det_keypoints[:, 1].clamp_(min=0, max=img_shape[0])

        if rescale:
            assert img_meta.get('scale_factor') is not None
            det_keypoints /= det_keypoints.new_tensor(img_meta['scale_factor'])[:2].unsqueeze(0)
        
        return scores, det_labels, det_keypoints, bbox_index

    def loss(self, 
         hidden_states: Tensor,
         references: List[Tensor],
         memory_text: Tensor, 
         text_token_mask: Tensor,
         memory_relation_text: Tensor,
         relation_text_token_mask: Tensor,
         enc_outputs_class: Tensor, 
         enc_outputs_coord: Tensor,
         batch_data_samples: SampleList, 
         memory: Tensor,
         memory_mask: Tensor,
         spatial_shapes: Tensor,
         level_start_index: Tensor,
         valid_ratios: Tensor,
         dn_meta: Dict[str, int]) -> dict:
        
        batch_gt_instances = []
        batch_img_metas = []
        for data_sample in batch_data_samples:
            batch_img_metas.append(data_sample.metainfo)
            batch_gt_instances.append(data_sample.gt_instances)

        all_layers_outputs_classes, all_layers_outputs_coords = self.forward(hidden_states, references, memory_text, text_token_mask)
        self.text_masks = text_token_mask

        # Set bbox size to 0
        all_layers_outputs_coords = torch.cat([
            all_layers_outputs_coords[..., :2],
            torch.zeros_like(all_layers_outputs_coords[..., 2:4])
        ], dim=-1)
        
        losses = self.loss_by_feat(all_layers_cls_scores=all_layers_outputs_classes,
                                   all_layers_bbox_preds=all_layers_outputs_coords,
                                   enc_cls_scores=enc_outputs_class,
                                   enc_bbox_preds=enc_outputs_coord,
                                   batch_gt_instances=batch_gt_instances,
                                   batch_img_metas=batch_img_metas,
                                   dn_meta=dn_meta)

        # ---- early return if no relations in the whole batch ----
        # Condition: each sample either lacks relation_matrices attribute or has zero relation types.
        no_relations_batch = all(
            (not hasattr(ds.gt_instances, 'relation_matrices')) or
            ds.gt_instances.relation_matrices.numel() == 0 or
            ds.gt_instances.relation_matrices.shape[-1] == 0
            for ds in batch_data_samples
        )
        # Also safeguard if relation text tokens are zero-length.
        if no_relations_batch or memory_relation_text.shape[1] == 0:
            return losses
        # --------------------------------------------------------

        outputs_classes = all_layers_outputs_classes[-1].detach()
        outputs_coords = all_layers_outputs_coords[-1].detach()
        hidden_states = hidden_states[-1]
        references = references[-1].detach()

        relation_targets, relation_weights = \
            self.get_relation_targets(
                outputs_classes, 
                outputs_coords, 
                batch_data_samples,
                len_rel_text=memory_relation_text.shape[1]
            )

        relation_mask = relation_weights > 0
        relation_mask = torch.any(relation_mask, dim=2) | torch.any(relation_mask, dim=1)

        relation_preds = self.relation_branch.forward(
            hidden_states=hidden_states,
            memory=memory,
            memory_mask=memory_mask,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            references=references,
            memory_relation_text=memory_relation_text,
            relation_text_token_mask=relation_text_token_mask,
            mask=relation_mask,
        ) # (batch_size, num_decoder_layer, num_queries, num_queries, len_text)

        # Loss is not computed for the padded regions of the text.
        assert (relation_text_token_mask.dim() == 2)
        text_mask = (relation_text_token_mask > 0).unsqueeze(1).unsqueeze(1)
        text_mask = text_mask.repeat(1, relation_targets.size(1), relation_targets.size(2), 1)

        relation_targets = torch.masked_select(relation_targets, text_mask)

        relation_weights = relation_weights[..., None].repeat(1, 1, 1, text_mask.size(-1))
        relation_weights = torch.masked_select(relation_weights, text_mask)

        # Remove where weight is not >0
        keep = relation_weights > 0
        relation_targets = relation_targets[keep]
        relation_weights = relation_weights[keep]

        if relation_weights.numel() == 0:
            return losses

        if isinstance(self.loss_relation, QualityFocalLoss):
            raise NotImplementedError('QualityFocalLoss for GroundingDINOHead is not supported yet.')
        
        for decoder_layer in range(relation_preds.shape[1]):
            relation_preds_layer = relation_preds[:, decoder_layer, :, :]
            relation_preds_layer = torch.masked_select(relation_preds_layer, text_mask)
            relation_preds_layer = relation_preds_layer[keep]

            loss_relation_layer = self.loss_relation(relation_preds_layer, relation_targets, relation_weights)

            if decoder_layer == relation_preds.shape[1] - 1:
                losses['loss_relation'] = loss_relation_layer
            else:
                losses[f'd{decoder_layer}.loss_relation'] = loss_relation_layer        
        return losses
    
    def loss_by_feat(
        self,
        all_layers_cls_scores: Tensor,
        all_layers_bbox_preds: Tensor,
        enc_cls_scores: Tensor,
        enc_bbox_preds: Tensor,
        batch_gt_instances: InstanceList,
        batch_img_metas: List[dict],
        dn_meta: Dict[str, int],
        batch_gt_instances_ignore: Optional[InstanceList] = None
    ) -> Dict[str, Tensor]:
        """Loss function.

        Args:
            all_layers_cls_scores (Tensor): Classification scores of all
                decoder layers, has shape (num_decoder_layers, bs,
                num_queries_total, cls_out_channels), where
                `num_queries_total` is the sum of `num_denoising_queries`
                and `num_matching_queries`.
            all_layers_bbox_preds (Tensor): Regression outputs of all decoder
                layers. Each is a 4D-tensor with normalized coordinate format
                (cx, cy, w, h) and has shape (num_decoder_layers, bs,
                num_queries_total, 4).
            enc_cls_scores (Tensor): The score of each point on encode
                feature map, has shape (bs, num_feat_points, cls_out_channels).
            enc_bbox_preds (Tensor): The proposal generate from the encode
                feature map, has shape (bs, num_feat_points, 4) with the last
                dimension arranged as (cx, cy, w, h).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            dn_meta (Dict[str, int]): The dictionary saves information about
                group collation, including 'num_denoising_queries' and
                'num_denoising_groups'. It will be used for split outputs of
                denoising and matching parts and loss calculation.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        (all_layers_cls_scores, all_layers_bbox_preds, _, _) = \
            self.split_outputs(
                all_layers_cls_scores, all_layers_bbox_preds, dn_meta)

        assert batch_gt_instances_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            'for batch_gt_instances_ignore setting to None.'

        losses_cls, losses_bbox = multi_apply(
            self.loss_by_feat_single,
            all_layers_cls_scores,
            all_layers_bbox_preds,
            batch_gt_instances=batch_gt_instances,
            batch_img_metas=batch_img_metas)

        loss_dict = dict()
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i in \
                zip(losses_cls[:-1], losses_bbox[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            num_dec_layer += 1

        if enc_cls_scores is not None:
            enc_loss_cls, enc_losses_bbox = \
                self.loss_by_feat_single(
                    enc_cls_scores, enc_bbox_preds,
                    batch_gt_instances=batch_gt_instances,
                    batch_img_metas=batch_img_metas)
            loss_dict['enc_loss_cls'] = enc_loss_cls
            loss_dict['enc_loss_bbox'] = enc_losses_bbox

        return loss_dict
    
    def loss_by_feat_single(self, cls_scores: Tensor, bbox_preds: Tensor,
                            batch_gt_instances: InstanceList,
                            batch_img_metas: List[dict]) -> Tuple[Tensor]:
        """Loss function for outputs from a single decoder layer of a single
        feature level.

        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images, has shape (bs, num_queries, cls_out_channels).
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape (bs, num_queries, 4).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.

        Returns:
            Tuple[Tensor]: A tuple including `loss_cls` and `loss_box`
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        with torch.no_grad():
            cls_reg_targets = self.get_targets(cls_scores_list,
                                               bbox_preds_list,
                                               batch_gt_instances,
                                               batch_img_metas)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        labels = torch.stack(labels_list, 0)
        label_weights = torch.stack(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        # ===== this change =====
        # Loss is not computed for the padded regions of the text.
        assert (self.text_masks.dim() == 2)
        text_masks = self.text_masks.new_zeros(
            (self.text_masks.size(0), self.max_text_len))
        text_masks[:, :self.text_masks.size(1)] = self.text_masks
        text_mask = (text_masks > 0).unsqueeze(1)
        text_mask = text_mask.repeat(1, cls_scores.size(1), 1)
        cls_scores = torch.masked_select(cls_scores, text_mask).contiguous()

        labels = torch.masked_select(labels, text_mask)
        label_weights = label_weights[...,
                                      None].repeat(1, 1, text_mask.size(-1))
        label_weights = torch.masked_select(label_weights, text_mask)

        # classification loss
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        if isinstance(self.loss_cls, QualityFocalLoss):
            raise NotImplementedError(
                'QualityFocalLoss for GroundingDINOHead is not supported yet.')
        else:
            loss_cls = self.loss_cls(
                cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes across all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        bbox_preds = bbox_preds.reshape(-1, 4)

        # regression L1 loss
        loss_bbox = self.loss_bbox(
            bbox_preds, bbox_targets, bbox_weights, avg_factor=num_total_pos)
        return loss_cls, loss_bbox
    
    def get_relation_targets(self, 
                        cls_scores: Tensor, 
                        bbox_preds: Tensor, 
                        batch_data_samples: SampleList,
                        len_rel_text: int) -> Tuple[Tensor, Tensor, int, int]:
        """Get relation targets for the relation branch.

        Returns:
            relation_target (Tensor): Relation targets of shape (batch_size, num_queries, num_queries, len_rel_text)
            relation_weights (Tensor): Relation weights of shape (batch_size, num_queries, num_queries)
        """
        batch_size = cls_scores.shape[0]

        relation_target_list, relation_weight_list = \
            multi_apply(self.get_relation_targets_single,
                [cls_scores[i] for i in range(batch_size)],
                [bbox_preds[i] for i in range(batch_size)],
                batch_data_samples,
                len_rel_text=len_rel_text
            )
        relation_targets = torch.stack(relation_target_list, dim=0)
        relation_weights = torch.stack(relation_weight_list, dim=0)
        return relation_targets, relation_weights

    def get_relation_targets_single(self,
                                     cls_scores: Tensor,
                                     bbox_preds: Tensor,
                                     data_sample: InstanceData,
                                     len_rel_text: int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Get relation targets for a single image.

        Returns:
            relation_targets (Tensor): Relation targets of shape (num_queries, num_queries, len_rel_text)
            relation_weights (Tensor): Relation weights of shape (num_queries, num_queries)
        """
        pos_inds, pos_assigned_gt_inds = self.get_query_to_gt_assignments(
            cls_scores, bbox_preds, data_sample
        )
        num_queries = cls_scores.shape[0]

        positive_map_relation = data_sample.gt_positive_map_relation    # (num_relations, max_text_len)
        positive_map_relation = positive_map_relation[:, :len_rel_text] # (num_relations, len_rel_text)

        gt_relation_matrices = data_sample.gt_instances.relation_matrices  # (num_gt, num_gt, num_relations)
        src_indices, dst_indices, type_indices = gt_relation_matrices.nonzero(as_tuple=False).unbind(1)

        # Shape: (num_gt, num_gt, len_rel_text)
        # [i, j, k] if relation i->j is associated with token k
        gt_positive_map = torch.zeros(
            (gt_relation_matrices.shape[0], gt_relation_matrices.shape[0], len_rel_text),
            dtype=torch.float32, device=gt_relation_matrices.device)
        gt_positive_map[src_indices, dst_indices, :] = positive_map_relation[type_indices]
        
        relation_targets = torch.zeros(num_queries, num_queries, len_rel_text, dtype=torch.float32, device=cls_scores.device)
        relation_weights = torch.zeros(num_queries, num_queries, dtype=torch.float32, device=cls_scores.device)
        
        if pos_inds.numel() > 0:
            relation_targets[pos_inds[:, None], pos_inds] = \
                gt_positive_map[pos_assigned_gt_inds[:, None], pos_assigned_gt_inds, :]
            relation_weights[pos_inds[:, None], pos_inds] = 1.0

        return relation_targets, relation_weights
    
    def get_query_to_gt_assignments(self,
                           cls_scores: Tensor,
                           bbox_preds: Tensor,
                           data_sample: InstanceData) -> Tuple[Tensor, Tensor, Tensor]:
        """Get query to ground truth assignments for a single image.

        Args:
            cls_scores (Tensor): Classification scores [num_queries, cls_out_channels]
            bbox_preds (Tensor): Box predictions [num_queries, 4]
            data_sample (InstanceData): Data sample for a single image

        Returns:
            Tuple[Tensor, Tensor, Tensor]: (pos_inds, pos_assigned_gt_inds),
                where pos_inds are the positive query indices,
                pos_assigned_gt_inds are the assigned ground truth indices.
        """
        img_h, img_w = data_sample.metainfo['img_shape']
        factor = bbox_preds.new_tensor([img_w, img_h, img_w, img_h]).unsqueeze(0)
        bbox_preds_scaled = bbox_cxcywh_to_xyxy(bbox_preds) * factor

        assign_result = self.assigner.assign(
            pred_instances=InstanceData(scores=cls_scores, bboxes=bbox_preds_scaled),
            gt_instances=data_sample.gt_instances,
            img_meta=data_sample.metainfo)

        pos_inds = torch.nonzero(assign_result.gt_inds > 0, as_tuple=False).squeeze(-1).unique()

        if pos_inds.numel() > 0:
            pos_assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1
        else:
            pos_assigned_gt_inds = pos_inds.new_empty((0,), dtype=torch.long)

        return pos_inds, pos_assigned_gt_inds