from __future__ import unicode_literals, print_function, division

import sys
import os
import json
import torch
import torch.nn as nn
from torch.autograd import Variable

import torch.nn.functional as F
sys.path.append('../')
from src.model.vqa import UniterForVisualSummEnc
from src.utils.const import BUCKET_SIZE, IMG_DIM
from src.utils.beam import Beam
from torch.nn import LayerNorm, Linear
from src.nat_model.build_decoder import get_decoder
from src.nat_model import levenshtein_utils as utils
import numpy as np

def init_lstm_wt(config, lstm):
    for names in lstm._all_weights:
        for name in names:
            if name.startswith('weight_'):
                wt = getattr(lstm, name)
                wt.data.uniform_(-config.rand_unif_init_mag, config.rand_unif_init_mag)
            elif name.startswith('bias_'):
                # set forget bias to 1
                bias = getattr(lstm, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data.fill_(0.)
                bias.data[start:end].fill_(1.)

def init_wt_normal(config, wt):
    wt.data.normal_(std=config.trunc_norm_init_std)

def init_linear_wt(config, linear):
    linear.weight.data.normal_(std=config.trunc_norm_init_std)
    if linear.bias is not None:
        linear.bias.data.normal_(std=config.trunc_norm_init_std)

class Pooler(nn.Module):
    def __init__(self, hidden_size):
        super(Pooler, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        pooled_output = self.dense(hidden_states)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class ImgPooler(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ImgPooler, self).__init__()
        self.dense = nn.Linear(input_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        pooled_output = self.dense(hidden_states)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class Attention(nn.Module):
    def __init__(self, config, input_dim, hidden_dim):
        super(Attention, self).__init__()
        # attention
        self.config = config
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.decode_proj = nn.Linear(self.input_dim, self.hidden_dim)
        self.v = nn.Linear(self.hidden_dim, 1, bias=False)
        init_linear_wt(config, self.decode_proj)
        init_linear_wt(config, self.v)



    def forward(self, s_t_hat, encoder_outputs, encoder_feature,coverage):
        b, t_k, n = list(encoder_outputs.size())

        dec_fea = self.decode_proj(s_t_hat)  # B x 2*hidden_dim
        dec_fea_expanded = dec_fea.unsqueeze(1).expand(b, t_k, n).contiguous()  # B x t_k x 2*hidden_dim
        dec_fea_expanded = dec_fea_expanded.view(-1, n)  # B * t_k x 2*hidden_dim

        att_features = encoder_feature + dec_fea_expanded  # B * t_k x 2*hidden_dim

        e = F.tanh(att_features)  # B * t_k x 2*hidden_dim
        scores = self.v(e)  # B * t_k x 1
        scores = scores.view(-1, t_k)  # B x t_k
        # print("Internal-Shape-enc pad mask:{}  enc_out:{}".format(enc_padding_mask.shape,
        #                                                  encoder_outputs.shape))
        attn_dist_ = F.softmax(scores, dim=1)
        normalization_factor = attn_dist_.sum(1, keepdim=True)
        attn_dist = attn_dist_ / normalization_factor

        attn_dist = attn_dist.unsqueeze(1)  # B x 1 x t_k
        c_t = torch.bmm(attn_dist, encoder_outputs)  # B x 1 x n
        c_t = c_t.view(b, -1)  # B x 2*hidden_dim

        attn_dist = attn_dist.view(-1, t_k)  # B x t_k


        return c_t, attn_dist,coverage

class GRUDecoder(nn.Module):
    def __init__(self, config, vocab_size,embedding_dim,hidden_dim):
        super(GRUDecoder, self).__init__()
        self.config = config
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.attention_network = Attention(config, config.hidden_dim, config.hidden_dim)
        # decoder
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        init_wt_normal(config, self.embedding.weight)

        self.x_context = nn.Linear(self.hidden_dim + self.embedding_dim, self.embedding_dim)
        self.W_h = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)

        self.GRU_layer = nn.GRU(self.embedding_dim,self.hidden_dim,batch_first=True)

        init_lstm_wt(config, self.GRU_layer)


        # p_vocab
        self.out1 = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.out2 = nn.Linear(self.hidden_dim, self.vocab_size)
        init_linear_wt(config, self.out2)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, y_t_1,
                s_t_0,
                s_t_1,
                encoder_outputs,
                c_t_1,
                coverage,
                step):
        y_t_1_embd = self.embedding(y_t_1)
        y_t_1_embd = y_t_1_embd.detach()

        y_t_1_embd = self.dropout(y_t_1_embd)
        x = self.x_context(torch.cat((c_t_1, y_t_1_embd), 1))
        self.GRU_layer.flatten_parameters()
        lstm_out, s_t = self.GRU_layer(x.unsqueeze(1), s_t_1)

        # h_decoder, c_decoder = s_t
        # s_t_hat = torch.cat((h_decoder.view(-1, self.hidden_dim),
        #                      c_decoder.view(-1, self.hidden_dim)), 1)  # B x 2*hidden_dim
        encoder_feature = encoder_outputs.view(-1, self.hidden_dim)
        encoder_feature = self.W_h(encoder_feature)
        s_t_hat=s_t.squeeze(0)
        c_t, attn_dist ,coverage_next = self.attention_network(s_t_hat, encoder_outputs, encoder_feature,
                                                               coverage)


        if step > 0:
            coverage = coverage_next

        output = torch.cat((lstm_out.view(-1, self.hidden_dim), c_t), 1)  # B x hidden_dim * 3
        output = self.out1(output)  # B x hidden_dim

        # output = F.relu(output)

        output = self.out2(output)  # B x vocab_size
        vocab_dist = F.softmax(output, dim=1)

        final_dist = vocab_dist

        return final_dist, s_t, c_t, attn_dist,coverage

class ResidualBlock(nn.Module):

    def __init__(self, layer, d_model, d_hidden, drop_ratio, pos=0):
        super().__init__()
        self.layer = layer
        self.dropout = nn.Dropout(drop_ratio)
        self.layernorm = LayerNorm(d_model)
        self.pos = pos

    def forward(self, *x):
        return self.layernorm(x[self.pos] + self.dropout(self.layer(*x)))

class FeedForward(nn.Module):

    def __init__(self, d_model, d_hidden):
        super().__init__()
        self.linear1 = Linear(d_model, d_hidden)
        self.linear2 = Linear(d_hidden, d_model)

    def forward(self, x):
        return self.linear2(F.relu(self.linear1(x)))

def weight_init(m):
    if isinstance(m, nn.Linear):
        m.weight.data.normal_(std=0.0001)
        # nn.init.xavier_normal_(m.weight)
        # nn.init.constant_(m.bias, 0)
    # 也可以判断是否为conv2d，使用相应的初始化方式
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
     # 是否为批归一化层
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 0)
    elif isinstance(m, nn.BatchNorm2d):#FusedLayerNorm
        nn.init.constant_(m.weight, 0)
        # nn.init.constant_(m.bias, 0)

class MultiModal(nn.Module):
    def __init__(self, config, tokenizer, model_file_path=None, is_eval=False):
        super(MultiModal, self).__init__()
        self.config = config
        vocab_size=len(tokenizer.vocab)
        meta = json.load(open(config.meta_file, 'r'))
        self.cls_ = meta['CLS']
        self.sep = meta['SEP']
        self.bos = 101
        self.unk = 100
        self.eos = 102
        self.pad = 0
        self.tokenizer = tokenizer
        self.beam_size = 1
        ckpt_file = config.checkpoint
        checkpoint = torch.load(ckpt_file)
        self.property_gen()
        self.text_encoder = UniterForVisualSummEnc.from_pretrained(
            config.model_config, checkpoint, img_dim=IMG_DIM)

        # self.decoder=GRUDecoder(config, vocab_size,embedding_dim,hidden_dim)
        self.decoder_nonauto = get_decoder(tokenizer.vocab)
        self.decoder_nonauto.decoder.apply(weight_init)
        # shared the embedding between encoder and decoder
        # self.decoder.embedding.weight = self.text_encoder.uniter.embeddings.word_embeddings.weight
        if is_eval:
            self.text_encoder = self.text_encoder.eval()
            self.decoder_nonauto = self.decoder_nonauto.eval()


        if model_file_path is not None:
            self.setup_train(model_file_path)

        if config.score_ref:
            self.txt_pooler = Pooler(config.hidden_dim)
            self.txt_img_pooler = Pooler(config.hidden_dim)
            self.out_pooler = Pooler(config.hidden_dim)
            init_linear_wt(config, self.txt_pooler.dense)
            init_linear_wt(config, self.txt_img_pooler.dense)
            init_linear_wt(config, self.out_pooler.dense)
            self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        if config.decoder_init and config.decoder_init_methods=='global':
            self.img_pooler = ImgPooler(2048, config.hidden_dim)
        elif config.decoder_init and config.decoder_init_methods=='softlb':
            self.img_pooler = ImgPooler(1601, config.hidden_dim)


    def setup_train(self, model_file_path=None):
        start_iter, start_loss = 0,0
        if model_file_path is not None:
            state = torch.load(model_file_path, map_location= lambda storage, location: storage)
            # start_iter = state['iter']
            # start_loss = state['current_loss']
            if "encoder_state_dict" in state:
                self.text_encoder.load_state_dict(state['encoder_state_dict'], strict=False)
            self.decoder_nonauto.load_state_dict(state['decoder_state_dict'], strict=False)
        return start_iter, start_loss

    def forward(self, batch, encoder_training=False):
        if self.config.guiding:
            batch['img_feat'] = batch['img_feat'] * batch['img_useful']
        encoder_output, pooled_output = self.text_encoder(batch, training=encoder_training)
        dec_batch = batch['dec_batch']
        max_dec_len = max(batch['dec_len'])
        b, t_k, n = list(encoder_output.size())
        s_t_1 = pooled_output.unsqueeze(0)
        if self.config.decoder_init and self.config.decoder_init_methods=='global':
            s_t_1 = self.get_decoder_init(s_t_1, batch['img_feat'], batch['img_useful'])
        elif self.config.decoder_init and self.config.decoder_init_methods=='softlb':
            s_t_1 = self.get_decoder_init_softlabels(s_t_1, batch['soft_labels'], batch['img_useful'])
        s_t_0 = s_t_1.unsqueeze(0)
        c_t = Variable(torch.zeros((b, self.config.hidden_dim)))
        coverage = Variable(torch.zeros((b, t_k)))
        c_t = c_t.to(device = encoder_output.device)
        coverage = coverage.to(device = encoder_output.device)
        final_dists = []
        attn_dists = []
        Ias = []
        for di in range(min(max_dec_len, self.config.max_dec_steps)):
            y_t_1 = dec_batch[:, di]  # Teacher forcing
            final_dist, s_t_1, c_t, attn_dist , next_coverage = self.decoder(y_t_1,
                                                          s_t_0,
                                                          s_t_1,
                                                          encoder_output,
                                                          c_t,
                                                          coverage,
                                                          di)
            final_dists.append(final_dist)
            attn_dists.append(attn_dist)
        if self.config.score_ref:
            sim_loss = self.score_ref_loss(encoder_output, pooled_output, s_t_1.squeeze(0),batch['txt_lens'], batch['img_useful'])
        else:
            sim_loss = None
        return torch.stack(final_dists, dim=1), attn_dists, coverage, Ias,sim_loss

    def forward_nonauto(self, batch, libnat_cuda, encoder_training=False):
        if self.config.guiding:
            batch['img_feat'] = batch['img_feat'] * batch['img_useful']
        encoder_output, pooled_output, embedding_output, attn_masks = self.text_encoder(batch, training=encoder_training)
        attn_masks = attn_masks==0
        encoder_outs = {
                        "encoder_out": encoder_output,
                        "encoder_padding_mask": attn_masks,
                        "encoder_embedding": embedding_output,
                        "encoder_states": None,
                        "src_tokens": None,
                        "src_lengths": None
                        }
        encoder_outs["encoder_out"] = encoder_outs["encoder_out"].transpose(0,1)
        tgt_tokens = batch['dec_batch']
        d1 = torch.tensor([i for i in range(tgt_tokens.shape[0])])
        d2 = (tgt_tokens!=0).sum(1)
        shape_1 = tgt_tokens.shape[1]
        d2 = d2-d2.eq(shape_1).long() #防止出现某个数字维度超过上限
        tgt_tokens[d1,d2] = self.sep
        # print("mms_nonauto.py 311 tgt_tokens: ",tgt_tokens)
        prev_output_tokens = self.decoder_nonauto.inject_noise(tgt_tokens)
        outputs = self.decoder_nonauto(encoder_outs, prev_output_tokens, tgt_tokens, libnat_cuda)
        return outputs

    def score_ref_loss(self, encoder_output,pooled_output,s_t_end, txt_lens, img_useful):
        bs =  encoder_output.shape[0]
        text_pooled = []
        for i in range(bs):
            txt_len = txt_lens[i]
            cur_pool = self.text_encoder.uniter.pooler(encoder_output[i:i+1,:txt_len,:])
            text_pooled.append(cur_pool)
        text_pooled = torch.stack(text_pooled,dim=0)

        emb_text = self.txt_pooler(text_pooled)
        emb_img_txt = self.txt_img_pooler(pooled_output)
        emb_out = self.out_pooler(s_t_end)
        sc_txt = self.cos(emb_text.squeeze(1), emb_out)
        sc_img_txt =self.cos(emb_img_txt, emb_out)

        img_use = img_useful[:,0,0]
        ones=torch.ones_like(img_use).float()
        neg_ones = -ones
        zeros = torch.zeros_like(ones).float()
        sim_loss_w = torch.where(img_use>=1, ones, neg_ones)
        sim_loss = -sim_loss_w*sc_img_txt
        sim_loss = torch.where(sim_loss>=0, sim_loss,zeros)

        return sim_loss

    def get_decoder_init(self, s_t_1, img_feat, img_useful):
        # 'global method init the s_t_0'
        img_pooled = self.img_pooler(img_feat)
        img_pooled = torch.mean(img_pooled, dim=1)
        img_pooled_use = (img_pooled.unsqueeze(1)*img_useful).transpose(0,1)
        _s_t_1 = img_pooled_use + s_t_1
        return _s_t_1

    def get_decoder_init_softlabels(self, s_t_1, soft_labels, img_useful):
        img_pooled = self.img_pooler(soft_labels)
        img_pooled = torch.mean(img_pooled, dim=1)
        img_pooled_use = (img_pooled.unsqueeze(1) * img_useful).transpose(0, 1)
        _s_t_1 = img_pooled_use + s_t_1
        return _s_t_1

    def decode(self, batch):
        if self.config.guiding:
            batch['img_feat'] = batch['img_feat'] * batch['img_useful']

        encoder_output, pooled_output = self.text_encoder(batch)
        max_dec_len = max(batch['dec_len'])
        b, t_k, n = list(encoder_output.size())
        s_t_1 = pooled_output.unsqueeze(0)
        if self.config.decoder_init and self.config.decoder_init_methods == 'global':
            s_t_1 = self.get_decoder_init(s_t_1, batch['img_feat'], batch['img_useful'])
        elif self.config.decoder_init and self.config.decoder_init_methods == 'softlb':
            s_t_1 = self.get_decoder_init_softlabels(s_t_1, batch['soft_labels'], batch['img_useful'])
        s_t_0 = s_t_1.unsqueeze(0)
        c_t = Variable(torch.zeros((b, self.config.hidden_dim)))
        coverage = Variable(torch.zeros((b, t_k)))
        c_t = c_t.to(device = encoder_output.device)
        coverage = coverage.to(device = encoder_output.device)
        final_dists = []
        attn_dists = []
        Ias = []
        latest_tokens = [self.cls_ for _ in range(b)]
        for di in range(min(max_dec_len, self.config.max_dec_steps)):
            y_t_1 = Variable(torch.LongTensor(latest_tokens))
            y_t_1 = y_t_1.to(encoder_output.device)
            final_dist, s_t_1, c_t, attn_dist , next_coverage = self.decoder_nonauto(y_t_1,
                                                          s_t_0,
                                                          s_t_1,
                                                          encoder_output,
                                                          c_t,
                                                          coverage,
                                                          di)
            log_probs = torch.log(final_dist)
            topk_log_probs, topk_ids = torch.topk(log_probs, 1)
            latest_tokens = [topk_ids[i][0] for i in range(b)]

            final_dists.append(final_dist)
            attn_dists.append(attn_dist)
        if self.config.score_ref:
            sim_loss = self.score_ref_loss(encoder_output, pooled_output, s_t_1.squeeze(0),batch['txt_lens'], batch['img_useful'])
        else:
            sim_loss = None
        return torch.stack(final_dists, dim=1), attn_dists, coverage, Ias,sim_loss

    @property
    def allow_length_beam(self):
        return True

    def property_gen(self):
       self.eos_penalty = 0.0
       self.max_iter = 10
       self.max_ratio = 2
       self.decoding_format = None
       self.retain_dropout = False
       self.adaptive = True
       self.retain_history = False
       self.reranking = False

    def generate(self, batch, prefix_tokens=None, constraints=None):
        reranker = None
        # TODO: better encoder inputs?
        src_tokens = batch["input_ids"]
        bsz, src_len = src_tokens.size()
        encoder_output, pooled_output, embedding_output, attn_masks = self.text_encoder(batch)
        attn_masks = attn_masks == 0
        encoder_outs = {
            "encoder_out": encoder_output,
            "encoder_padding_mask": attn_masks,
            "encoder_embedding": embedding_output,
            "encoder_states": None,
            "src_tokens": None,
            "src_lengths": None
        }
        # initialize
        # encoder_out = model.forward_encoder([src_tokens, src_lengths])
        encoder_outs["encoder_out"] = encoder_outs["encoder_out"].transpose(0, 1)
        prev_decoder_out = self.decoder_nonauto.initialize_output_tokens(encoder_outs, src_tokens)

        if self.beam_size > 1:
            assert (
                self.allow_length_beam
            ), "{} does not support decoding with length beam.".format(
                self.__class__.__name__
            )

            # regenerate data based on length-beam
            length_beam_order = (
                utils.new_arange(src_tokens, self.beam_size, bsz).t().reshape(-1)
            )
            encoder_outs = utils.reorder_encoder_out(
                encoder_outs, length_beam_order
            )
            prev_decoder_out = utils.regenerate_length_beam(
                prev_decoder_out, self.beam_size, self.pad, self.unk, self.bos, self.eos)
            bsz = bsz * self.beam_size

        sent_idxs = torch.arange(bsz)
        prev_output_tokens = prev_decoder_out.output_tokens.clone()

        if self.retain_history:
            prev_decoder_out = prev_decoder_out._replace(history=[prev_output_tokens])
        finalized = [[] for _ in range(bsz)]

        def is_a_loop(x, y, s, a):
            b, l_x, l_y = x.size(0), x.size(1), y.size(1)
            if l_x > l_y:
                y = torch.cat([y, x.new_zeros(b, l_x - l_y).fill_(self.pad)], 1)
                s = torch.cat([s, s.new_zeros(b, l_x - l_y)], 1)
                if a is not None:
                    a = torch.cat([a, a.new_zeros(b, l_x - l_y, a.size(2))], 1)
            elif l_x < l_y:
                x = torch.cat([x, y.new_zeros(b, l_y - l_x).fill_(self.pad)], 1)
            return (x == y).all(1), y, s, a

        def finalized_hypos(step, prev_out_token, prev_out_score, prev_out_attn):
            cutoff = prev_out_token.ne(self.pad)
            tokens = prev_out_token[cutoff]
            if prev_out_score is None:
                scores, score = None, None
            else:
                scores = prev_out_score[cutoff]
                score = scores.mean()

            if prev_out_attn is None:
                hypo_attn, alignment = None, None
            else:
                hypo_attn = prev_out_attn[cutoff]
                alignment = hypo_attn.max(dim=1)[1]
            return {
                "steps": step,
                "tokens": tokens,
                "positional_scores": scores,
                "score": score,
                "hypo_attn": hypo_attn,
                "alignment": alignment,
            }


        for step in range(self.max_iter + 1):

            decoder_options = {
                "eos_penalty": self.eos_penalty,
                "max_ratio": self.max_ratio,
                "decoding_format": self.decoding_format,
            }
            prev_decoder_out = prev_decoder_out._replace(
                step=step,
                max_step=self.max_iter + 1,
            )

            decoder_out, batch_entr = self.decoder_nonauto.forward_decoder(
                prev_decoder_out, encoder_outs, **decoder_options)
            if step==0:
                first_batch_entr=batch_entr

            if self.adaptive:
                # terminate if there is a loop
                terminated, out_tokens, out_scores, out_attn = is_a_loop(
                    prev_output_tokens,
                    decoder_out.output_tokens,
                    decoder_out.output_scores,
                    decoder_out.attn,
                )
                decoder_out = decoder_out._replace(
                    output_tokens=out_tokens,
                    output_scores=out_scores,
                    attn=out_attn,
                )

            else:
                terminated = decoder_out.output_tokens.new_zeros(
                    decoder_out.output_tokens.size(0)
                ).bool()

            if step == self.max_iter:  # reach last iteration, terminate
                terminated.fill_(1)

            # collect finalized sentences
            finalized_idxs = sent_idxs[terminated]
            finalized_tokens = decoder_out.output_tokens[terminated]
            finalized_scores = decoder_out.output_scores[terminated]
            finalized_attn = (
                None
                if (decoder_out.attn is None or decoder_out.attn.size(0) == 0)
                else decoder_out.attn[terminated]
            )

            if self.retain_history:
                finalized_history_tokens = [h[terminated] for h in decoder_out.history]

            for i in range(finalized_idxs.size(0)):
                finalized[finalized_idxs[i]] = [
                    finalized_hypos(
                        step,
                        finalized_tokens[i],
                        finalized_scores[i],
                        None if finalized_attn is None else finalized_attn[i],
                    )
                ]

                if self.retain_history:
                    finalized[finalized_idxs[i]][0]["history"] = []
                    for j in range(len(finalized_history_tokens)):
                        finalized[finalized_idxs[i]][0]["history"].append(
                            finalized_hypos(
                                step, finalized_history_tokens[j][i], None, None
                            )
                        )

            # check if all terminated
            if terminated.sum() == terminated.size(0):
                break

            # for next step
            not_terminated = ~terminated
            prev_decoder_out = decoder_out._replace(
                output_tokens=decoder_out.output_tokens[not_terminated],
                output_scores=decoder_out.output_scores[not_terminated],
                attn=decoder_out.attn[not_terminated]
                if (decoder_out.attn is not None and decoder_out.attn.size(0) > 0)
                else None,
                history=[h[not_terminated] for h in decoder_out.history]
                if decoder_out.history is not None
                else None,
            )
            encoder_outs = utils.reorder_encoder_out(
                encoder_outs, not_terminated.nonzero(as_tuple=False).squeeze()
            )
            sent_idxs = sent_idxs[not_terminated]
            prev_output_tokens = prev_decoder_out.output_tokens.clone()

        if self.beam_size > 1:
            if reranker is not None:
                finalized = self.rerank(
                    reranker, finalized, [src_tokens, src_lengths], self.beam_size
                )

            # aggregate information from length beam
            finalized = [
                finalized[
                    np.argmax(
                        [
                            finalized[self.beam_size * i + j][0]["score"]
                            for j in range(self.beam_size)
                        ]
                    )
                    + self.beam_size * i
                    ]
                for i in range(len(finalized) // self.beam_size)
            ]

        return finalized, first_batch_entr

    def sort_beams(self, beams):
        return sorted(beams, key=lambda h: h.avg_log_prob, reverse=True)

    def beam_search(self, batch):
        encoder_output, pooled_output = self.text_encoder(batch)
        final_dists = self.decoder_nonauto(encoder_output)
        topk_log_probs, topk_ids = torch.topk(final_dists, 1)
        topk_ids = topk_ids[0,:self.config.max_dec_steps,0]
        topk_ids = [x.item() for x in topk_ids]
        return topk_ids

