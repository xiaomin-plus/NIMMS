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

class MultiModal(nn.Module):
    def __init__(self, config, tokenizer, model_file_path=None, is_eval=False):
        super(MultiModal, self).__init__()
        self.config = config
        vocab_size=len(tokenizer.vocab)
        meta = json.load(open(config.meta_file, 'r'))
        self.cls_ = meta['CLS']
        self.sep = meta['SEP']
        self.tokenizer = tokenizer
        embedding_dim= config.hidden_dim
        hidden_dim= config.hidden_dim

        ckpt_file = config.checkpoint
        checkpoint = torch.load(ckpt_file)
        self.text_encoder = UniterForVisualSummEnc.from_pretrained(
            config.model_config, checkpoint, img_dim=IMG_DIM)
        self.text_encoder.eval()

        self.decoder=GRUDecoder(config, vocab_size,embedding_dim,hidden_dim)

        # shared the embedding between encoder and decoder
        self.decoder.embedding.weight = self.text_encoder.uniter.embeddings.word_embeddings.weight
        if is_eval:
            self.text_encoder = self.text_encoder.eval()
            self.decoder = self.decoder.eval()


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
            self.decoder.load_state_dict(state['decoder_state_dict'], strict=False)
        return start_iter, start_loss

    def forward(self, batch, encoder_training=False):
        if self.config.guiding:
            batch['img_feat'] = batch['img_feat'] * batch['img_useful']
        encoder_output, pooled_output, embedding_output, attn_masks, all_attention_scores = self.text_encoder(batch, training=encoder_training)
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

        encoder_output, pooled_output, embedding_output, attn_masks, all_attention_scores = self.text_encoder(batch)
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
            final_dist, s_t_1, c_t, attn_dist , next_coverage = self.decoder(y_t_1,
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

    def sort_beams(self, beams):
        return sorted(beams, key=lambda h: h.avg_log_prob, reverse=True)

    def beam_search(self, batch):
        if self.config.guiding:
            batch['img_feat'] = batch['img_feat'] * batch['img_useful']

        encoder_output, pooled_output, embedding_output, attn_masks, all_attention_scores = self.text_encoder(batch)
        b, t_k, n = list(encoder_output.size())
        encoder_output = encoder_output.expand(self.config.beam_size, t_k, n).contiguous()
        s_t_1 = pooled_output.unsqueeze(0)
        if self.config.decoder_init and self.config.decoder_init_methods == 'global':
            s_t_1 = self.get_decoder_init(s_t_1, batch['img_feat'], batch['img_useful'])
        elif self.config.decoder_init and self.config.decoder_init_methods == 'softlb':
            s_t_1 = self.get_decoder_init_softlabels(s_t_1, batch['soft_labels'], batch['img_useful'])
        s_t_0 = pooled_output
        c_t = Variable(torch.zeros((self.config.beam_size, self.config.hidden_dim)))
        coverage = Variable(torch.zeros((self.config.beam_size, t_k)))
        c_t = c_t.to(device=encoder_output.device)
        coverage = coverage.to(device=encoder_output.device)

        beams = [Beam(tokens=[self.cls_],
                      log_probs=[0.0],
                      state=s_t_1[:, 0],
                      context=c_t[0],
                      coverage=coverage[0])
                 for _ in range(self.config.beam_size)]
        results = []
        steps = 0

        while steps < self.config.max_dec_steps and len(results) < self.config.beam_size:
            latest_tokens = [h.latest_token for h in beams]
            # latest_tokens = [t if t < len(self.vocab) else self.vocab.stoi[self.config.UNKNOWN_TOKEN] \
            #                  for t in latest_tokens]
            y_t_1 = Variable(torch.LongTensor(latest_tokens))
            y_t_1 = y_t_1.to(encoder_output.device)
            all_state_h = []

            all_context = []

            for h in beams:
                state_h = h.state
                all_state_h.append(state_h)

                all_context.append(h.context)
            s_t_1 = torch.stack(all_state_h, 0).transpose(1, 0)
            c_t = torch.stack(all_context, 0)

            coverage_t_1 = None
            if self.config.is_coverage:
                all_coverage = []
                for h in beams:
                    all_coverage.append(h.coverage)
                coverage_t_1 = torch.stack(all_coverage, 0)

            final_dist, s_t_1, c_t, attn_dist , next_coverage = self.decoder(y_t_1,
                                                          s_t_0,
                                                          s_t_1,
                                                          encoder_output,
                                                          c_t,
                                                          coverage,
                                                          steps)

            log_probs = torch.log(final_dist)
            topk_log_probs, topk_ids = torch.topk(log_probs, self.config.beam_size * 2)

            dec_h = s_t_1
            dec_h = dec_h.squeeze()

            all_beams = []
            num_orig_beams = 1 if steps == 0 else len(beams)
            for i in range(num_orig_beams):
                h = beams[i]
                state_i = (dec_h[i].unsqueeze(0))
                context_i = c_t[i]
                # coverage_i = next_coverage[i] if next_coverage else None
                coverage_i = next_coverage[i]

                for j in range(self.config.beam_size * 2):  # for each of the top 2*beam_size hyps:
                    new_beam = h.extend(token=topk_ids[i, j].item(),
                                        log_prob=topk_log_probs[i, j].item(),
                                        state=state_i,
                                        context=context_i,
                                        coverage=coverage_i)
                    all_beams.append(new_beam)

            beams = []
            for h in self.sort_beams(all_beams):
                if h.latest_token == self.sep:
                    if steps >= self.config.min_dec_steps:
                        results.append(h)
                else:
                    beams.append(h)
                if len(beams) == self.config.beam_size or len(results) == self.config.beam_size:
                    break

            steps += 1

        if len(results) == 0:
            results = beams

        beams_sorted = self.sort_beams(results)

        return beams_sorted[0], all_attention_scores

