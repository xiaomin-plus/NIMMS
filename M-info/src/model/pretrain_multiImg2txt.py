"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

UNITER for pretraining
"""
from collections import defaultdict

import torch
from torch import nn
from torch.nn import functional as F
from apex.normalization.fused_layer_norm import FusedLayerNorm as LayerNorm

from .layer import GELU, BertOnlyMLMHead
from .model import UniterModel, UniterPreTrainedModel
from .ot import optimal_transport_dist
import random
import time
import math

class RegionFeatureRegression(nn.Module):
    " for MRM"
    def __init__(self, hidden_size, feat_dim, img_linear_weight):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                 GELU(),
                                 LayerNorm(hidden_size, eps=1e-12))

        self.weight = img_linear_weight
        self.bias = nn.Parameter(torch.zeros(feat_dim))

    def forward(self, input_):
        hidden = self.net(input_)
        output = F.linear(hidden, self.weight.t(), self.bias)
        return output


class RegionClassification(nn.Module):
    " for MRC(-kl)"
    def __init__(self, hidden_size, label_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                 GELU(),
                                 LayerNorm(hidden_size, eps=1e-12),
                                 nn.Linear(hidden_size, label_dim))

    def forward(self, input_):
        output = self.net(input_)
        return output

class UniterForPretraining(UniterPreTrainedModel):
    """ UNITER pretraining """
    def __init__(self, config, img_dim, img_label_dim):
        super().__init__(config)
        self.uniter = UniterModel(config, img_dim)
        self.cls = BertOnlyMLMHead(
            config, self.uniter.embeddings.word_embeddings.weight)
        self.feat_regress = RegionFeatureRegression(
            config.hidden_size, img_dim,
            self.uniter.img_embeddings.img_linear.weight)
        self.region_classifier = RegionClassification(
            config.hidden_size, img_label_dim)
        self.itm_output = nn.Linear(config.hidden_size, 2)
        self.apply(self.init_weights)
        self.BCEloss = nn.BCELoss()
    
    def forward_paralle(self, batch, compute_loss=False):
        batch = defaultdict(lambda: None, batch)
        input_ids = batch['input_ids']
        position_ids = batch['position_ids']
        img_feat = batch['img_feat']
        img_pos_feat = batch['img_pos_feat']
        attention_mask = batch['attn_masks']
        gather_index = batch['gather_index']
        txt_labels = batch['txt_labels']
        img_mask_tgt = batch['img_mask_tgt']
        img_masks = batch['img_masks']
        distribution = torch.zeros(1, 28996).to(device=input_ids.device)
        inp_len = input_ids.shape[1]

        text_w = 0.5
        img_w = 0.5
        if img_feat!=None:
            r_text = text_w/(inp_len-2)
            r_img = img_w/(img_masks.shape[1])
        else:
            r_text = 1/(inp_len-2)
            r_img = 0

        # 1. text mask agrregate
        _, input_len = input_ids.shape
        paralle_dim = 50
        img_paralle_dim = 50
        if img_feat!= None:
            _, img_len, img_dim = img_feat.shape
        else:
            img_len = 0
        _, attn_len = attention_mask.shape
        
        def get_txt_paralle_batch(start_pos, paralle_dim, sum_len):
            cur_input_ids = input_ids.repeat(paralle_dim, 1)
            cur_position_ids = position_ids.repeat(paralle_dim, 1)
            cur_txt_labels = txt_labels.repeat(paralle_dim, 1)
            if img_feat!=None:
                cur_img_feat = img_feat.repeat(paralle_dim, 1, 1)
                cur_img_pos_feat = img_pos_feat.repeat(paralle_dim, 1, 1)
            else:
                cur_img_feat, cur_img_pos_feat = None, None
            cur_attention_mask = attention_mask.repeat(paralle_dim, 1)
            cur_gather_index = gather_index.repeat(paralle_dim, 1)
            if start_pos + paralle_dim > sum_len:
                pos_seq = [j for j in range(start_pos, sum_len)]
            else:
                pos_seq = [j for j in range(start_pos, start_pos+paralle_dim)]
            for i, j in enumerate(pos_seq):
                cur_txt_labels[i, j-1] = input_ids[0, j]
                cur_input_ids[i,j] = 103
            return cur_input_ids, cur_position_ids, cur_txt_labels, cur_img_feat, cur_img_pos_feat, cur_attention_mask, cur_gather_index
        def get_img_paralle_batch(start_pos, img_paralle_dim, sum_len, inp_len):
            cur_input_ids = input_ids.repeat(img_paralle_dim, 1)
            cur_position_ids = position_ids.repeat(img_paralle_dim, 1)
            cur_txt_labels = txt_labels.repeat(img_paralle_dim, 1)
            cur_img_feat = img_feat.repeat(img_paralle_dim, 1, 1)
            cur_img_pos_feat = img_pos_feat.repeat(img_paralle_dim, 1, 1)
            cur_attention_mask = attention_mask.repeat(img_paralle_dim, 1)
            cur_gather_index = gather_index.repeat(img_paralle_dim, 1)
            cur_img_masks = img_masks.repeat(img_paralle_dim, 1)
            cur_img_mask_tgt = img_mask_tgt.repeat(img_paralle_dim, 1)
            if start_pos + paralle_dim > sum_len:
                pos_seq = [j for j in range(start_pos, sum_len)]
            else:
                pos_seq = [j for j in range(start_pos, start_pos+paralle_dim)]
            for i, j in enumerate(pos_seq):
                cur_img_masks[i,j] = True
                cur_img_mask_tgt[i,inp_len+j] = 1
            return cur_input_ids, cur_position_ids, cur_txt_labels, cur_img_feat, cur_img_pos_feat, cur_attention_mask, cur_gather_index, cur_img_masks, cur_img_mask_tgt
        # import pdb
        # pdb.set_trace()
        start_pos = 1
        turn = math.ceil((input_len-1)/paralle_dim)
        # import pdb
        # pdb.set_trace()
        for i in range(turn):
            cur_input_ids, cur_position_ids, cur_txt_labels, cur_img_feat, cur_img_pos_feat, cur_attention_mask, cur_gather_index = get_txt_paralle_batch(start_pos, paralle_dim, input_len)
            prediction_score = self.forward_mlm(cur_input_ids, cur_position_ids,
                                cur_img_feat, cur_img_pos_feat,
                                cur_attention_mask, cur_gather_index,
                                cur_txt_labels, compute_loss)
            # print("time: ", time.time()-s1)
            start_pos = start_pos + paralle_dim
            distribution = distribution + (r_text*torch.nn.functional.softmax(prediction_score, -1)).sum(dim=0)
        # import pdb
        # pdb.set_trace()
        # 2. image mask agrregate
        distribution_img = torch.zeros(1, 28996).to(device=input_ids.device)
        if r_img !=0:
            start_pos = 0
            turn = math.ceil(img_len/paralle_dim)
            for i in range(turn):
                # print(start_pos, i)
                cur_input_ids, cur_position_ids, cur_txt_labels, cur_img_feat, cur_img_pos_feat, cur_attention_mask, cur_gather_index, cur_img_masks, cur_img_mask_tgt = get_img_paralle_batch(start_pos, img_paralle_dim, img_len, inp_len)
                prediction_score = self.forward_mrfr(cur_input_ids, cur_position_ids,
                                        cur_img_feat, cur_img_pos_feat,
                                        cur_attention_mask, cur_gather_index,
                                        cur_img_masks, cur_img_mask_tgt,
                                        None, compute_loss)
                # prediction_score[0][0] = -1000
                start_pos = start_pos + paralle_dim
                top100_v = prediction_score.topk(k=20,dim=1).values[:,-1]
                top100_v_dim1 = top100_v.shape[0]
                top100_v = top100_v.unsqueeze(1).expand(top100_v_dim1, 28996)
                lower_than_100 = ((prediction_score-top100_v)<0)*-1000
                prediction_score = prediction_score + lower_than_100
                distribution_img = distribution_img + (r_img*torch.nn.functional.softmax(prediction_score, -1)).sum(dim=0)
                

        distribution = distribution + distribution_img
        # import pdb
        # pdb.set_trace()
        return distribution
    
    def forward(self, batch, compute_loss=False):
        batch = defaultdict(lambda: None, batch)
        input_ids = batch['input_ids']
        position_ids = batch['position_ids']
        img_feat = batch['img_feat']
        img_pos_feat = batch['img_pos_feat']
        attention_mask = batch['attn_masks']
        gather_index = batch['gather_index']
        txt_labels = batch['txt_labels']
        img_mask_tgt = batch['img_mask_tgt']
        img_masks = batch['img_masks']
        distribution = torch.zeros(1, 28996).to(device=input_ids.device)
        inp_len = input_ids.shape[1]

        text_w = 0.5
        img_w = 0.5
        if img_feat!=None:
            r_text = text_w/(inp_len-2)
            r_img = img_w/(img_masks.shape[1])
        else:
            r_text = 1/(inp_len-2)
            r_img = 0

        # 1. text mask agrregate
        for i in range(input_ids.shape[1]):
            if i==0:
                continue
            else:
                if i>1:
                    input_ids[0,i-1] = token_id
                token_id = input_ids[0,i].clone()
                txt_labels[0,i-1] = -1
                txt_labels[0,i] = token_id
                input_ids[0,i] = 103
                prediction_score = self.forward_mlm(input_ids, position_ids,
                                    img_feat, img_pos_feat,
                                    attention_mask, gather_index,
                                    txt_labels, compute_loss)
                distribution = distribution + r_text*torch.nn.functional.softmax(prediction_score)
        # 2. image mask agrregate
        input_ids[0,i] = token_id
        distribution_img = torch.zeros(1, 28996).to(device=input_ids.device)
        if r_img !=0:
            for i in range(img_masks.shape[1]):
                if i>0:
                    img_masks[0,i-1] = False
                    img_mask_tgt[0,inp_len+i-1] = 0
                img_masks[0,i] = True
                img_mask_tgt[0,inp_len+i] = 1
                # import pdb
                # pdb.set_trace()
                prediction_score = self.forward_mrfr(input_ids, position_ids,
                                        img_feat, img_pos_feat,
                                        attention_mask, gather_index,
                                        img_masks, img_mask_tgt,
                                        None, compute_loss)
                # prediction_score[0][0] = -1000
                top100_v = prediction_score.topk(k=20).values[0][-1]
                lower_than_100 = (prediction_score<top100_v)*-1000
                prediction_score = prediction_score + lower_than_100
                distribution_img = distribution_img + r_img*torch.nn.functional.softmax(prediction_score)
                
        # import pdb
        # pdb.set_trace()
        distribution = distribution + distribution_img
        # import pdb
        # pdb.set_trace()
        return distribution
    
    def forward_mlm(self, input_ids, position_ids, img_feat, img_pos_feat,
                    attention_mask, gather_index,
                    txt_labels, compute_loss=True):
        sequence_output = self.uniter(input_ids, position_ids,
                                      img_feat, img_pos_feat,
                                      attention_mask, gather_index,
                                      output_all_encoded_layers=False)
        sequence_output = sequence_output.detach()
        # get only the text part
        sequence_output = sequence_output[:, :input_ids.size(1), :]
        # only compute masked tokens for better efficiency
        # import pdb
        # pdb.set_trace()
        masked_output = self._compute_masked_hidden(sequence_output,
                                                    txt_labels != -1)
        prediction_scores = self.cls(masked_output)

        if compute_loss:
            masked_lm_loss = F.cross_entropy(prediction_scores,
                                             txt_labels[txt_labels != -1],
                                             reduction='none')
            return masked_lm_loss
        else:
            return prediction_scores
        
    def forward_mrm(self, input_ids, position_ids, img_feat, img_pos_feat,
                    attention_mask, gather_index, txt_labels, dec_labels, tf_weight, compute_loss=True):
        # import pdb
        # pdb.set_trace()

        sequence_output = self.uniter(input_ids, position_ids,
                                      img_feat, img_pos_feat,
                                      attention_mask, gather_index,
                                      output_all_encoded_layers=False)
        sequence_output = sequence_output.detach()
        # get only the text part
        # sequence_output = sequence_output[:, :input_ids.size(1), :]
        # only compute masked tokens for better efficiency
        masked_output = self._compute_masked_hidden(sequence_output,
                                                    txt_labels != -1)
        prediction_scores = self.cls(masked_output)
        dec_labels = dec_labels[:,1:]
        dec_labels_onehot = F.one_hot(dec_labels, num_classes=28996)
        dec_labels_onehot = dec_labels_onehot.sum(dim=1).float()
        dec_labels_onehot[:,0] = 0
        dec_lens = dec_labels_onehot.sum(dim=1)
        # import pdb
        # pdb.set_trace()
        if compute_loss:
            prediction_scores = F.log_softmax(prediction_scores,1)
            masked_lm_loss = prediction_scores * dec_labels_onehot * tf_weight
            masked_lm_loss = -masked_lm_loss.sum(dim=1)/dec_lens
            # masked_lm_loss = masked_lm_loss.mean()
            return masked_lm_loss
        else:
            return prediction_scores

    def _compute_masked_hidden(self, hidden, mask):
        """ get only the masked region (don't compute unnecessary hiddens) """
        mask = mask.unsqueeze(-1).expand_as(hidden)
        hidden_masked = hidden[mask].contiguous().view(-1, hidden.size(-1))
        return hidden_masked

    def forward_mrfr(self, input_ids, position_ids, img_feat, img_pos_feat,
                     attention_mask, gather_index, img_masks, img_mask_tgt,
                     feat_targets, compute_loss=True):
        sequence_output = self.uniter(input_ids, position_ids,
                                      img_feat, img_pos_feat,
                                      attention_mask, gather_index,
                                      output_all_encoded_layers=False,
                                      img_masks=img_masks)

        # only compute masked tokens for better efficiency
        masked_output = self._compute_masked_hidden(sequence_output,
                                                    img_mask_tgt)
        prediction_feat = self.cls(masked_output)

        return prediction_feat

    def forward_mrfr_origin(self, input_ids, position_ids, img_feat, img_pos_feat,
                     attention_mask, gather_index, img_masks, img_mask_tgt,
                     feat_targets, compute_loss=True):
        sequence_output = self.uniter(input_ids, position_ids,
                                      img_feat, img_pos_feat,
                                      attention_mask, gather_index,
                                      output_all_encoded_layers=False,
                                      img_masks=img_masks)

        # only compute masked tokens for better efficiency
        masked_output = self._compute_masked_hidden(sequence_output,
                                                    img_mask_tgt)
        prediction_feat = self.feat_regress(masked_output)

        if compute_loss:
            mrfr_loss = F.mse_loss(prediction_feat, feat_targets,
                                   reduction='none')
            return mrfr_loss
        else:
            return prediction_feat

    def forward_itm(self, input_ids, position_ids, img_feat, img_pos_feat,
                    attention_mask, gather_index, targets, ot_inputs,
                    compute_loss=True):
        sequence_output = self.uniter(input_ids, position_ids,
                                      img_feat, img_pos_feat,
                                      attention_mask, gather_index,
                                      output_all_encoded_layers=False)
        pooled_output = self.uniter.pooler(sequence_output)
        pooled_output = pooled_output.detach()
        itm_scores = self.itm_output(pooled_output)

        # OT loss
        if ot_inputs is not None:
            ot_scatter = ot_inputs['ot_scatter']

            b = sequence_output.size(0)
            tl = input_ids.size(1)
            il = img_feat.size(1)
            max_l = max(ot_inputs['scatter_max'] + 1, tl+il)

            ot_scatter = ot_scatter.unsqueeze(-1).expand_as(sequence_output)
            ctx_emb = torch.zeros(b, max_l, self.config.hidden_size,
                                  dtype=sequence_output.dtype,
                                  device=sequence_output.device
                                  ).scatter_(dim=1, index=ot_scatter,
                                             src=sequence_output)
            txt_emb = ctx_emb[:, :tl, :]
            img_emb = ctx_emb[:, tl:tl+il, :]

            txt_pad = ot_inputs['txt_pad']
            img_pad = ot_inputs['img_pad']
            # NOTE: run in fp32 for stability
            ot_dist = optimal_transport_dist(txt_emb.float(), img_emb.float(),
                                             txt_pad, img_pad).to(txt_emb)
            ot_pos_dist = ot_dist.masked_select(targets == 1)
            ot_neg_dist = ot_dist.masked_select(targets == 0)
            ot_loss = (ot_pos_dist, ot_neg_dist)
        else:
            ot_loss = None

        if compute_loss:
            itm_loss = F.cross_entropy(itm_scores, targets, reduction='none')
            return itm_loss, ot_loss
        else:
            return itm_scores, ot_dist

    def forward_mrc(self, input_ids, position_ids, img_feat, img_pos_feat,
                    attention_mask, gather_index, img_masks, img_mask_tgt,
                    label_targets, task, compute_loss=True):
        sequence_output = self.uniter(input_ids, position_ids,
                                      img_feat, img_pos_feat,
                                      attention_mask, gather_index,
                                      output_all_encoded_layers=False,
                                      img_masks=img_masks)

        # only compute masked regions for better efficiency
        masked_output = self._compute_masked_hidden(sequence_output,
                                                    img_mask_tgt)
        prediction_soft_label = self.region_classifier(masked_output)

        if compute_loss:
            if "kl" in task:
                prediction_soft_label = F.log_softmax(
                    prediction_soft_label, dim=-1)
                mrc_loss = F.kl_div(
                    prediction_soft_label, label_targets, reduction='none')
            else:
                # background class should not be the target
                label_targets = torch.max(label_targets[:, 1:], dim=-1)[1] + 1
                mrc_loss = F.cross_entropy(
                    prediction_soft_label, label_targets,
                    ignore_index=0, reduction='none')
            return mrc_loss
        else:
            return prediction_soft_label
