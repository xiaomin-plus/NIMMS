"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.
"""
from collections import defaultdict

from torch import nn
from torch.nn import functional as F
from apex.normalization.fused_layer_norm import FusedLayerNorm as LayerNorm

from .layer import GELU
from .model_drop import UniterPreTrainedModel, UniterModel

class UniterForVisualSummEnc(UniterPreTrainedModel):
    """ Finetune UNITER for Multi-modal Summarizaiton Encoder
    """
    def __init__(self, config, img_dim):
        super().__init__(config)
        self.uniter = UniterModel(config, img_dim)
        self.apply(self.init_weights)

    def forward(self, batch, training=False):
        batch = defaultdict(lambda: None, batch)
        input_ids = batch['input_ids']
        position_ids = batch['position_ids']
        img_feat = batch['img_feat']
        img_pos_feat = batch['img_pos_feat']
        attn_masks = batch['attn_masks']
        gather_index = batch['gather_index']
        attn_vis_cross = batch['attn_vis_cross']
        attn_txt_all = batch['attn_txt_all']
        sequence_output, embedding_output, all_attention_scores,visual_outputs = self.uniter(input_ids, position_ids,
                                      img_feat, img_pos_feat,
                                      attn_masks, attn_vis_cross, attn_txt_all, gather_index,
                                      output_all_encoded_layers=False)
        pooled_output = self.uniter.pooler(sequence_output)

        if training==False:
            sequence_output = sequence_output.detach()
            pooled_output = pooled_output.detach()

        return sequence_output, pooled_output, embedding_output, attn_masks, all_attention_scores,visual_outputs
