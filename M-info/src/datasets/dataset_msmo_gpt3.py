import glob

from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
import torch
import os
import pickle
import spacy
eng_model = spacy.load('en_core_web_sm')
import numpy as np
import math
from numpy import random
random.seed(10)
import json
import argparse
import sys
import csv
from pytorch_pretrained_bert import BertTokenizer
from torch.nn.utils.rnn import pad_sequence
from toolz.sandbox import unzip

sys.path.append('../')
from src.datasets.convert_imgdir import load_npz
from src.datasets.data import pad_tensors, get_gather_index
from src.datasets.sampler import TokenBucketSampler
from src.datasets.loader import PrefetchLoader
from src.configs import base_config
from src.utils.utils import merge_sub_word
from src.utils import const
from src.utils import structure_parse
from stanfordcorenlp import StanfordCoreNLP

csv.field_size_limit(sys.maxsize)

FIELDNAMES = ['image_id', 'image_w', 'image_h', 'num_boxes', 'boxes', 'features']

# device_id = config.device_id

# torch.manual_seed(123)

def shuffle_image(img_lens, image_useful, random_seed):
    random.seed(random_seed)
    shuffle_image_proj = {}
    random_index = [str(i+1) for i in range(img_lens) if str(i+1)+'.jpg' not in image_useful]
    random.shuffle(random_index)
    for index in range(img_lens):
        iname = str((index + 1)) + '.jpg'
        if iname in image_useful:
            shuffle_image_proj[str((index + 1))] = str((index + 1))
        else:
            shuffle_image_proj[str((index + 1))] = random_index.pop()
    return shuffle_image_proj

def sample_balance_useless(img_lens, image_useful, random_seed):
    random.seed(random_seed)
    useless_index = [i for i in range(img_lens) if str(i + 1) + '.jpg' not in image_useful]
    useful_index = [int(img_str.split('.')[0])-1 for img_str in image_useful]
    random.shuffle(useless_index)
    useless_index = useless_index[:len(useful_index)]

    balance_index = useful_index + useless_index
    random.shuffle(balance_index)
    sample_image_proj = {ind:origin_ind for ind, origin_ind in enumerate(balance_index)}
    return sample_image_proj

def _compute_ot_scatter(txt_lens, max_txt_len, joint_len):
    ot_scatter = torch.arange(0, joint_len, dtype=torch.long
                              ).unsqueeze(0).repeat(len(txt_lens), 1)
    for i, tl in enumerate(txt_lens):
        max_ind = max_txt_len + (joint_len-tl)
        ot_scatter.data[i, tl:] = torch.arange(max_txt_len, max_ind,
                                               dtype=torch.long).data
    return ot_scatter


def _compute_pad(lens, max_len):
    pad = torch.zeros(len(lens), max_len, dtype=torch.uint8)
    for i, l in enumerate(lens):
        pad.data[i, l:].fill_(1)
    return pad

def random_word(tokens, vocab_range, mask):
    """
    Masking some random tokens for Language Model task with probabilities as in
        the original BERT paper.
    :param tokens: list of int, tokenized sentence.
    :param vocab_range: for choosing a random word
    :return: (list of int, list of int), masked tokens and related labels for
        LM prediction
    """
    output_label = []

    for i, token in enumerate(tokens):
        prob = random.random()
        # mask token with 15% probability
        if prob < 0.15:
            prob /= 0.15

            # 80% randomly change token to mask token
            if prob < 0.8:
                tokens[i] = mask

            # 10% randomly change token to random token
            elif prob < 0.9:
                tokens[i] = random.choice(list(range(*vocab_range)))

            # -> rest 10% randomly keep current token

            # append current token to output (we will predict these later)
            output_label.append(token)
        else:
            # no masking token (will be ignored by loss function later)
            output_label.append(-1)
    if all(o == -1 for o in output_label):
        # at least mask 1
        output_label[0] = tokens[0]
        tokens[0] = mask

    return tokens, output_label

CNAME = "bert"
class MultiDataset(Dataset):
    def __init__(self, config, images_dir, article_raw, tokenizer, image_useful_file,
                 image_useless_file, random_seed, data_mode='test', Imode = "D+Vs", balance_useful= False):
        self.config = config
        self.images_dir = images_dir
        self.data_mode = data_mode
        self.Imode = Imode

        # Init and Build vocab
        self.tokenizer = tokenizer
        self.random_seed = random_seed
        self.balance_useful = balance_useful
        self.start_num = 0
        meta = json.load(open(config.meta_file, 'r'))
        self.cls_ = meta['CLS']
        self.sep = meta['SEP']
        self.mask = meta['MASK']
        self.v_range = meta['v_range']
        self.sample_image_proj = None
        self.article_raw = article_raw
        example_id_path = self.images_dir + '/exID_{}_{}.pickle'.format(CNAME, data_mode)
        # new_summary_path = self.images_dir + '/{}_new_summary2000.txt'.format(data_mode)
        new_cap_path = self.images_dir + '/{}_new_cap_clip200.p'.format(data_mode)
        old_summary_minus1_path = self.images_dir + '/{}_old_summary2000_minus1.txt'.format(data_mode)
        # self.extract_article(article_raw, images_dir, data_mode)
        # with open(example_id_path, 'rb') as f:
        #     self.example_ids, self.txt_lens  = pickle.load(f)
        #     f.close()
        
        # with open(new_cap_path,'rb') as f:
        #     len_ex = len(self.example_ids)
        #     self.new_captions = pickle.load(f)
        #     # self.new_captions = self.new_captions + [""]*(len_ex-len(self.new_captions))
        #     f.close()
        #     self.new_example_ids = []

        with open(config.hyp_path,'r') as f:
            self.hyp_summs = f.readlines()
            f.close()

        # name2nbb_name = "/".join(images_dir.split('/')[:-1]) + "/name2nbb_{}.pkl".format(data_mode)
        txt2img_name = images_dir + "/txt2img_{}.pkl".format(data_mode)
        if not os.path.exists(txt2img_name):
            self.name2nbb, self.txt2img = self._get_name_to_nbb(self.images_dir, data_mode)
            # with open(name2nbb_name, 'wb') as fnb:
            #     pickle.dump(self.name2nbb, fnb)
            #     fnb.close()
            with open(txt2img_name, 'wb') as fnb:
                pickle.dump((self.name2nbb, self.txt2img), fnb)
                fnb.close()
        else:
            # with open(name2nbb_name, 'rb') as fnb:
            #     self.name2nbb = pickle.load(fnb)
            #     fnb.close()
            with open(txt2img_name, 'rb') as fnb:
                self.name2nbb, self.txt2img = pickle.load(fnb)
                fnb.close()
        # assert len(self.example_ids) == len(self.txt2img)
        # self.lens = [10 for _ in range(len(self.example_ids))]
        # self.lens = self.txt_lens
        with open("/data-yifan/data/meihuan2/dataset/MSMO_Finished/info-newSum-data/exID_test_gpt3_50.pickle", 'rb') as f:
            self.example_ids, self.txt_lens, self.new_captions = pickle.load(f)
            f.close()
        self.lens = [self.txt_lens[ex_id] + self.name2nbb[self.txt2img[ex]] for ex_id, ex in enumerate(self.example_ids)]

    def __len__(self):
        return len(self.lens)

    def create_mlm_io(self, input_ids):
        input_ids, txt_labels = random_word(input_ids,
                                            self.v_range,
                                            self.mask)
        input_ids = torch.tensor([self.cls_]
                                 + input_ids
                                 + [self.sep])
        txt_labels = torch.tensor([-1] + txt_labels + [-1])
        return input_ids, txt_labels

    def preprocess_doc(self,file_name, merge_title=True):
        f = open(file_name, 'r')
        text = f.read()
        f.close()
        text = text.lower()
        text = text.split('@body')
        title = text[0]
        text = text[-1].split('@summary')
        body = text[0]
        summary = text[1:]
        title = title.replace('\n', ' ')
        title = title.replace('@title', ' ')
        title = title.strip()

        body = body.replace('\n', ' ').strip()
        if merge_title:
            src_text = title + '. ' + body
        else:
            src_text = body

        summary = [s.replace('\n', ' ').strip() for s in summary]
        summary = ". ".join(summary) + '.'

        return src_text, summary

    def pgn_preprocess_doc(self, story_file):
        dm_single_close_quote = u'\u2019'  # unicode
        dm_double_close_quote = u'\u201d'
        END_TOKENS = ['.', '!', '?', '...', "'", "`", '"', dm_single_close_quote, dm_double_close_quote,
                      ")"]  # acceptable ways to end a sentence
        SENTENCE_START = '<s>'
        SENTENCE_END = '</s>'
        def read_text_file(text_file):
            lines = []
            with open(text_file, "r") as f:
                for line in f:
                    lines.append(line.strip())
            return lines

        def fix_missing_period(line):
            """Adds a period to a line that is missing a period"""
            if "@highlight" in line: return line
            if line == "": return line
            if line[-1] in END_TOKENS: return line
            # print line[-1]
            return line + " ."
        lines = read_text_file(story_file)

        # Lowercase everything
        lines = [line.lower() for line in lines]

        # Put periods on the ends of lines that are missing them (this is a problem in the dataset because many image captions don't end in periods; consequently they end up in the body of the article as run-on sentences)
        lines = [fix_missing_period(line) for line in lines]

        # Separate out article and abstract sentences
        article_lines = []
        highlights = []
        next_is_highlight = False
        for idx, line in enumerate(lines):
            if line == "":
                continue  # empty line
            elif line.startswith("@summary"):
                next_is_highlight = True
            elif next_is_highlight:
                highlights.append(line)
            else:
                article_lines.append(line)

        # Make article into a single string
        article = ' '.join(article_lines)

        # Make abstract into a signle string, putting <s> and </s> tags around the sentences
        abstract = ' '.join(highlights)

        return article, abstract

    def numerate_txt(self, title, summary):
        titles_pre = title.split(' ')
        titles_pre = titles_pre[:self.config.max_txt_len + 10]
        titles_pre = " ".join(titles_pre)
        inp = self.bert_tokenize(titles_pre.strip())
        if len(inp) > self.config.max_txt_len:
            inp = [self.cls_] + inp[:self.config.max_txt_len] + [self.sep]
        else:
            inp = [self.cls_] + inp + [self.sep]
        dec = self.bert_tokenize(summary.strip())
        # if self.config.key_w_loss:
        #     dec_pos, org_pos, dec_to_tokens = self.get_inp_pos(dec)
        # else:
        #     dec_pos = [0]
        return inp, len(inp), dec

    def _get_txt_feat(self,inp,dec):
        input_ids = torch.tensor(inp)
        # input_poses = self.input_poses[index]
        _dec_id = dec

        dec_inp, dec_tgt = self.get_dec_inp_targ_seqs(_dec_id,
                                                      self.config.max_dec_steps,
                                                      self.cls_,
                                                      self.sep)
        dec_len = len(dec_inp)
        dec_inp, dec_tgt = self.pad_decoder_inp_targ(self.config.max_dec_steps,
                                                     0,
                                                     dec_inp,
                                                     dec_tgt)
        dec_inp = torch.tensor(dec_inp)
        dec_tgt = torch.tensor(dec_tgt)
        dec_padding_mask = torch.ones((dec_len))
        dec_len = torch.tensor(dec_len)
        copy_position = torch.zeros(input_ids.shape[0])
        for i, x in enumerate(input_ids):
            if x in dec_tgt and x!=self.sep:
                copy_position[i] = 1
        return input_ids, dec_inp, dec_tgt, dec_padding_mask, dec_len, copy_position

    def merge_sub_word(self, sentence):
        sub_words = self.tokenizer.convert_ids_to_tokens(sentence)
        words = []
        i = 0
        len_sub = len(sub_words)
        cur_word = ''
        while i < len_sub:
            if sub_words[i].startswith('##'):
                cur_word = cur_word + sub_words[i][2:]
            else:
                if len(cur_word) != 0:
                    words.append(cur_word)
                cur_word = sub_words[i]
            i = i + 1
        if len(cur_word) != 0:
            words.append(cur_word)
        return words, sub_words

    def preprocess_txt(self, article_file, save_dir, nlp=None):
        txt_name = article_file.split('/')[-1].split('.')[0]
        dn = article_file.split('/')[-3]
        cur_save_path = save_dir + '/{}/article_{}/{}.pickle'.format(dn, CNAME, txt_name)
        src_text, summary = self.pgn_preprocess_doc(article_file)
        inp, inp_len, dec = self.numerate_txt(src_text, summary)
        # inp_to_tokens, sub_words = self.merge_sub_word(inp)
        # phrase_tensor = structure_parse.to_phrase(nlp, inp_to_tokens, sub_words)
        input_ids, \
        dec_batch, target_batch, dec_padding_mask, dec_len, \
        copy_position = self._get_txt_feat(inp, dec)

        if not os.path.exists(save_dir + '/{}/article_{}'.format(dn, CNAME)):
            os.mkdir(save_dir + '/{}/article_{}'.format(dn, CNAME))
        with open(cur_save_path, 'wb') as ft:
            pickle.dump((src_text, summary,
                         inp, inp_len, dec,
                         input_ids,
                         dec_batch, target_batch, dec_padding_mask, dec_len,
                         copy_position
                         # ,phrase_tensor
                         ), ft)
            ft.close()
        return inp_len

    def extract_article(self,article_raw, image_dir,data_mode,nlp=None):
        txt_lens = []
        if data_mode == 'train':
            article_pts = []
            for k in range(20):
                cur_pts = glob.glob(article_raw + '/data{}/article/*.txt'.format(k + 1))
                article_pts = article_pts + cur_pts
                for aid, article_file in enumerate(cur_pts):
                    if aid%1000==0:
                        print("extract_article: data{}-{}/{}".format(k+1, aid, len(cur_pts)))
                    inp_len = self.preprocess_txt(article_file, image_dir, nlp)
                    txt_lens.append(inp_len)
        else:
            article_pts = glob.glob(article_raw + '/{}_data/article/*.txt'.format(data_mode))
            for aid,article_file in enumerate(article_pts):
                if aid % 1000 == 0:
                    print("extract_article: {}/{}".format(aid, len(article_pts)))
                inp_len = self.preprocess_txt(article_file, image_dir, nlp)
                txt_lens.append(inp_len)

        example_ids = [pt.split('/')[-3] + '-' + pt.split('/')[-1].split('.')[0] for pt in article_pts]

        return example_ids, txt_lens

    def __getitem__(self, index: int):
        ex = self.example_ids[index]
        dn, ex_name = ex.split('-')
        img_name = self.txt2img[ex]
        article_file = self.article_raw + '/{}/article/{}.txt'.format(dn, ex_name)
        src_text, summary = self.pgn_preprocess_doc(article_file)
        src_text = src_text.lower()
        summary = summary.lower()
        
        dn, image_name = img_name.split('-')
        # txt_pickle_path = self.images_dir + '/{}/article_{}/{}.pickle'.format(dn, CNAME, txt_name)
        if self.Imode == "D+V":
            inp, inp_len, dec = self.numerate_txt(src_text, summary)
            # inp_to_tokens, sub_words = self.merge_sub_word(inp)
            # phrase_tensor = structure_parse.to_phrase(nlp, inp_to_tokens, sub_words)
            input_ids, \
            dec_batch, target_batch, dec_padding_mask, dec_len, \
            copy_position = self._get_txt_feat(inp, dec)

            # img_list = glob.glob(self.images_dir + '/{}/img_roi/{}*.npz'.format(dn, ex_name))
            img_pickle_path = self.images_dir + '/{}/img_roi/{}'.format(dn, image_name)
            # print("image_name: ", image_name)
            img_feat, img_pos_feat, num_bb, soft_labels = self._get_img_feat(img_pickle_path)
        elif self.Imode == "R+V":
            inp, inp_len, dec = self.numerate_txt(summary, "summary")
            # inp_to_tokens, sub_words = self.merge_sub_word(inp)
            # phrase_tensor = structure_parse.to_phrase(nlp, inp_to_tokens, sub_words)
            input_ids, \
            dec_batch, target_batch, dec_padding_mask, dec_len, \
            copy_position = self._get_txt_feat(inp, dec)

            # img_list = glob.glob(self.images_dir + '/{}/img_roi/{}*.npz'.format(dn, ex_name))
            img_pickle_path = self.images_dir + '/{}/img_roi/{}'.format(dn, image_name)
            # print("image_name: ", image_name)
            img_feat, img_pos_feat, num_bb, soft_labels = self._get_img_feat(img_pickle_path)
        elif self.Imode == "D+Vs":
            inp, inp_len, dec = self.numerate_txt(src_text, summary)
            # inp_to_tokens, sub_words = self.merge_sub_word(inp)
            # phrase_tensor = structure_parse.to_phrase(nlp, inp_to_tokens, sub_words)
            input_ids, \
            dec_batch, target_batch, dec_padding_mask, dec_len, \
            copy_position = self._get_txt_feat(inp, dec)

            img_list = glob.glob(self.images_dir + '/{}/img_roi/{}*.npz'.format(dn, ex_name))
            for i, img_pickle_path in enumerate(img_list):
                # img_pickle_path = self.images_dir + '/{}/img_roi/{}'.format(dn, image_name)
                if i == 0:
                    img_feat, img_pos_feat, num_bb, soft_labels = self._get_img_feat(img_pickle_path)
                else:
                    cur_img_feat, cur_img_pos_feat, cur_num_bb, cur_soft_labels = self._get_img_feat(img_pickle_path)
                    img_feat = torch.cat([img_feat, cur_img_feat], dim=0)
                    img_pos_feat = torch.cat([img_pos_feat, cur_img_pos_feat], dim=0)
                    num_bb = num_bb + cur_num_bb
                    soft_labels = np.concatenate((soft_labels, cur_soft_labels), axis=0)
        elif self.Imode == 'R':
            inp, inp_len, dec = self.numerate_txt(summary, "summary")
            # print("summary: ", summary)
            # inp_to_tokens, sub_words = self.merge_sub_word(inp)
            # phrase_tensor = structure_parse.to_phrase(nlp, inp_to_tokens, sub_words)
            input_ids, \
            dec_batch, target_batch, dec_padding_mask, dec_len, \
            copy_position = self._get_txt_feat(inp, dec)

            img_feat, img_pos_feat, num_bb, soft_labels = None, np.array([]), 0, np.array([])

        elif self.Imode == 'CR':
            # summary = "a large truck parked next to a metal barrier at an airport. The truck has a large amount of cargo on its back, which is covered in a plastic sheet." + summary
            # summary = self.new_summary[index].lower()
            summary = self.new_captions[index] + " " + summary
            # print("CR summary: ", summary)
            inp, inp_len, dec = self.numerate_txt(summary, "summary")
            # inp_to_tokens, sub_words = self.merge_sub_word(inp)
            # phrase_tensor = structure_parse.to_phrase(nlp, inp_to_tokens, sub_words)
            input_ids, \
            dec_batch, target_batch, dec_padding_mask, dec_len, \
            copy_position = self._get_txt_feat(inp, dec)

            img_feat, img_pos_feat, num_bb, soft_labels = None, np.array([]), 0, np.array([])
        
        elif self.Imode == 'hyp':
            # summary = "a large truck parked next to a metal barrier at an airport. The truck has a large amount of cargo on its back, which is covered in a plastic sheet." + summary
            # summary = self.new_summary[index].lower()
            summary = self.hyp_summs[index]
            # print("CR summary: ", summary)
            inp, inp_len, dec = self.numerate_txt(summary, "summary")
            # inp_to_tokens, sub_words = self.merge_sub_word(inp)
            # phrase_tensor = structure_parse.to_phrase(nlp, inp_to_tokens, sub_words)
            input_ids, \
            dec_batch, target_batch, dec_padding_mask, dec_len, \
            copy_position = self._get_txt_feat(inp, dec)

            img_feat, img_pos_feat, num_bb, soft_labels = None, np.array([]), 0, np.array([])
        
        elif self.Imode == 'Rn':
            # summary = "Lorry was seen travelling at 50 miles per hour on a road in Hubei , China. Owner covered the driver 's cabin after it was damaged in an accident.".lower()
            summary = self.old_summary_minus1[index]
            inp, inp_len, dec = self.numerate_txt(summary, "summary")
            # inp_to_tokens, sub_words = self.merge_sub_word(inp)
            # phrase_tensor = structure_parse.to_phrase(nlp, inp_to_tokens, sub_words)
            input_ids, \
            dec_batch, target_batch, dec_padding_mask, dec_len, \
            copy_position = self._get_txt_feat(inp, dec)

            img_feat, img_pos_feat, num_bb, soft_labels = None, np.array([]), 0, np.array([])
        elif self.Imode == 'Ry':
            def random_select(src_sents):
                cands = []
                for line in src_sents:
                    if len(line.split(' ')) > 7:
                        cands.append(line)
                    while len(cands)>10:
                        break
                random.shuffle(cands)
                return cands[0]
            src_sents = src_text.split('.')

            # summary = summary + "Almost every inch of the driver 's cabin was covered with tarpaulin , leaving only a tiny square to allow the driver to look out"
            summary = random_select(src_sents) + summary
            inp, inp_len, dec = self.numerate_txt(summary, "summary")
            # inp_to_tokens, sub_words = self.merge_sub_word(inp)
            # phrase_tensor = structure_parse.to_phrase(nlp, inp_to_tokens, sub_words)
            input_ids, \
            dec_batch, target_batch, dec_padding_mask, dec_len, \
            copy_position = self._get_txt_feat(inp, dec)

            img_feat, img_pos_feat, num_bb, soft_labels = None, np.array([]), 0, np.array([])
        elif self.Imode == 'DR':
            summary = summary + src_text
            inp, inp_len, dec = self.numerate_txt(summary, "summary")
            # inp_to_tokens, sub_words = self.merge_sub_word(inp)
            # phrase_tensor = structure_parse.to_phrase(nlp, inp_to_tokens, sub_words)
            input_ids, \
            dec_batch, target_batch, dec_padding_mask, dec_len, \
            copy_position = self._get_txt_feat(inp, dec)

            img_feat, img_pos_feat, num_bb, soft_labels = None, np.array([]), 0, np.array([])

        # if self.config.textonly:
        #     img_feat, img_pos_feat, num_bb, soft_labels = self._get_img_feat(self.config.avg_img_npz)
        # else:
        #     img_feat, img_pos_feat, num_bb, soft_labels = self._get_img_feat(img_pickle_path)
        input_str, tgt_str = src_text, summary
        # with open(txt_pickle_path, 'rb') as fr:
        #     (input_str, tgt_str,
        #      inp, inp_len, dec,
        #      input_ids,
        #      dec_batch, target_batch, dec_padding_mask, dec_len,
        #      copy_position
        #      ) = pickle.load(fr)
        #     fr.close()

        input_lens = inp_len + num_bb
        attn_masks = torch.ones(input_lens, dtype=torch.long)
        soft_labels = torch.tensor(soft_labels)

        txt_labels = torch.zeros(len(input_ids))-1
        # txt_labels[2] = input_ids[2]
        txt_labels = txt_labels.long()
        # input_ids[2] = self.mask

        img_mask = torch.zeros(num_bb)==1
        # img_mask[1]=True
        z = torch.zeros(inp_len, dtype=torch.uint8)
        img_mask_tgt = torch.cat([z, img_mask], dim=0)

        return index, input_lens, input_ids, txt_labels, img_mask, img_mask_tgt,\
               img_feat, soft_labels, img_pos_feat, attn_masks, \
               dec_batch, target_batch, dec_padding_mask, dec_len, None,\
               input_str, tgt_str


    def _get_img_feat(self, filename):
        name, dump, nbb = load_npz(self.config.conf_th,
                                   self.config.max_bb,
                                   self.config.min_bb,
                                   self.config.num_bb,
                                   filename)
        img_feat = dump['features']
        img_bb = dump['norm_bb']
        soft_labels = dump['soft_labels']

        img_feat = torch.tensor(img_feat[:nbb, :]).float()
        img_bb = torch.tensor(img_bb[:nbb, :]).float()

        img_bb = torch.cat([img_bb, img_bb[:, 4:5] * img_bb[:, 5:]], dim=-1)

        return img_feat, img_bb, nbb, soft_labels

    def _get_name_to_nbb(self, image_dir,data_mode):
        name2nbb = {}
        txt2img = {}
        if data_mode=='test' or data_mode=='valid':
            pts = glob.glob(image_dir + '/{}_data/img_roi/*.npz'.format(data_mode))
        else:
            pts = []
            for k in range(20):
                cur_pts = glob.glob(image_dir + '/data{}/img_roi/*.npz'.format(k+1))
                pts = pts + cur_pts
        for pid, pt in enumerate(pts):
            if pid%1000==0:
                print('extract img: {}/{}'.format(pid,len(pts)))
            name, dump, nbb = load_npz(self.config.conf_th,
                                       self.config.max_bb,
                                       self.config.min_bb,
                                       self.config.num_bb,
                                       pt)
            dn, name = pt.split('/')[-3], pt.split('/')[-1]
            name2nbb[dn+'-'+name] = nbb
            txt_name = name.split('_')[0]
            txt2img[dn+'-'+txt_name] = dn+'-'+name

        return name2nbb, txt2img

    def get_name_to_nbb(self, image_dir):
        txt2img = {}
        if self.data_mode=='test':
            pts = glob.glob(image_dir + '/test_data/img_roi/*.npz')
        elif self.data_mode=='valid':
            pts = glob.glob(image_dir + '/valid_data/img_roi/*.npz')
        else:
            pts = []
            for k in range(20):
                cur_pts = glob.glob(image_dir + '/data{}/img_roi/*.npz'.format(k+1))
                pts = pts + cur_pts
        for pt in pts:
            dn, name = pt.split('/')[-3], pt.split('/')[-1]
            txt_name = name.split('_')[0]
            txt2img[dn+'-'+txt_name] = dn+'-'+name
        return txt2img

    def _get_ids_and_lens(self, titles, summaris):
        assert len(titles) == len(summaris)
        lens = []
        input_ids = []
        dec_poses = []
        dec_ids = []
        for ind in range(len(titles)):
            titles_pre = titles[ind].split(' ')
            titles_pre = titles_pre[:self.config.max_txt_len+10]
            titles_pre = " ".join(titles_pre)
            inp = self.bert_tokenize(titles_pre.strip())
            if len(inp) > self.config.max_txt_len:
                inp = [self.cls_] + inp[:self.config.max_txt_len] + [self.sep]
            else:
                inp = [self.cls_] + inp + [self.sep]
            input_ids.append(inp)
            lens.append(len(inp))

            dec = self.bert_tokenize(summaris[ind].strip())
            if self.config.key_w_loss:
                dec_pos, org_pos, dec_to_tokens = self.get_inp_pos(dec)
            else:
                dec_pos = [0]
            dec_ids.append(dec)
            dec_poses.append(dec_pos)
        return input_ids, None, lens, dec_ids, dec_poses

    def get_inp_pos(self, inp):
        inp_to_tokens, sub_words = merge_sub_word(self.tokenizer, inp)
        org_pos = eng_model(" ".join(inp_to_tokens))
        org_pos = [w.pos_ for w in org_pos]
        cur_index = 0
        inp_pos = []
        for subw in sub_words:
            if subw.startswith('##'):
                inp_pos.append(org_pos[cur_index-1])
            else:
                inp_pos.append(org_pos[cur_index])
                cur_index = cur_index+1
        return inp_pos, org_pos, inp_to_tokens

    def get_inp_ef(self, ind, inp):
        inp_to_tokens, sub_words = merge_sub_word(self.tokenizer, inp)
        if len(self.mg_ef[ind])!=0:
            org_str, org_pos = self.mg_ef[ind]
        else:
            org_str, org_pos = [],[]
        cur_index = 0
        inp_pos = []
        for iter, subw in enumerate(sub_words):
            if cur_index>=len(org_pos):
                inp_pos.append(0)
                continue
            if subw.startswith('##') and subw.replace('##','') in org_str[cur_index - 1]:
                inp_pos.append(org_pos[cur_index - 1])
            elif subw in org_str[cur_index]:
                inp_pos.append(org_pos[cur_index])
                cur_index = cur_index + 1
            else:
                cur_index = cur_index + 1
        return inp_pos

    def bert_tokenize(self, text):
        ids = []
        for word in text.strip().split():
            ws = self.tokenizer.tokenize(word)
            if not ws:
                # some special char
                continue
            ids.extend(self.tokenizer.convert_tokens_to_ids(ws))
        return ids

    def get_dec_inp_targ_seqs(self, sequence, max_len, start_id, stop_id):
        inp = [start_id] + sequence[:]
        target = sequence[:]
        if len(inp) > max_len:  # truncate
            inp = inp[:max_len]
            target = target[:max_len]  # no end_token
        else:  # no truncation
            target.append(stop_id)  # end token
        assert len(inp) == len(target)
        return inp, target

    def pad_decoder_inp_targ(self, max_len, pad_id,numericalized_inp,numericalized_tgt):
        while len(numericalized_inp) < max_len:
            numericalized_inp.append(pad_id)
        while len(numericalized_tgt) < max_len:
            numericalized_tgt.append(pad_id)
        return numericalized_inp,numericalized_tgt

def vqa_eval_collate(inputs):
    def sorted_batch(inputs):
        (qids, input_lens, input_ids, txt_labels, input_poses, img_feats, soft_labels, img_pos_feats, attn_masks, dec_batch, target_batch, dec_padding_mask,
         dec_len, dec_pos_f
         ) = map(list, unzip(inputs))
        input_lens = torch.stack(input_lens, dim=0)
        sorted_input_lens = torch.argsort(input_lens)
        qids = [qids[i] for i in sorted_input_lens]
        input_ids = [input_ids[i] for i in sorted_input_lens]
        img_feats = [img_feats[i] for i in sorted_input_lens]
        img_pos_feats = [img_pos_feats[i] for i in sorted_input_lens]
        attn_masks = [attn_masks[i] for i in sorted_input_lens]
        dec_batch = [dec_batch[i] for i in sorted_input_lens]
        target_batch = [target_batch[i] for i in sorted_input_lens]
        dec_padding_mask = [dec_padding_mask[i] for i in sorted_input_lens]
        return qids, input_lens, input_ids, img_feats, img_pos_feats, attn_masks, dec_batch, target_batch, dec_padding_mask, dec_len

    # (qids, input_lens, input_ids, img_feats, img_pos_feats, attn_masks, dec_batch, target_batch, dec_padding_mask, dec_len
    #  ) = sorted_batch(inputs)


    (qids, input_lens, input_ids, txt_labels, img_mask, img_mask_tgt,
     img_feats, soft_labels, img_pos_feats, attn_masks,
     dec_batch, target_batch, dec_padding_mask,dec_len, dec_pos_f,
     input_str, tgt_str
     ) = map(list, unzip(inputs))
    txt_lens = [i.size(0) for i in input_ids]
    # print("len qids: ", len(qids))
    # print("len input_ids_lens: ", len(input_ids))
    # print("len txt_lens: ", len(txt_lens))

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    # copy_position = pad_sequence(copy_position, batch_first=True, padding_value=0)
    txt_labels = pad_sequence(txt_labels, batch_first=True, padding_value=-1)
    img_mask = pad_sequence(img_mask, batch_first=True, padding_value=0)
    img_mask_tgt = pad_sequence(img_mask_tgt, batch_first=True, padding_value=0)
    position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long
                                ).unsqueeze(0)
    attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0)
    # attn_masks_s0, attn_masks_s1 = attn_masks.shape
    # attn_masks = attn_masks.unsqueeze(1)
    # attn_masks = attn_masks.expand(attn_masks_s0, attn_masks_s1, attn_masks_s1)

    # if targets[0] is None:
    #     targets = None
    # else:
    #     targets = torch.stack(targets, dim=0)
    dec_batch = pad_sequence(dec_batch, batch_first=True, padding_value=0)
    targets = pad_sequence(target_batch, batch_first=True, padding_value=0)
    dec_padding_mask = pad_sequence(dec_padding_mask, batch_first=True, padding_value=0)
    # dec_pos_f = pad_sequence(dec_pos_f, batch_first=True, padding_value=0)
    soft_labels = pad_sequence(soft_labels, batch_first=True, padding_value=0)
    soft_labels = soft_labels.float()
    if img_feats[0]!=None:
        num_bbs = [f.size(0) for f in img_feats]
    else:
        num_bbs = [0 for f in img_feats]

    num_bbs = torch.tensor(num_bbs)
    if img_feats[0]!=None:
        img_feats = pad_tensors(img_feats, num_bbs)
        img_pos_feat = pad_tensors(img_pos_feats, num_bbs)
    else:
        img_feats = None
        img_pos_feat = None

    bs, max_tl = input_ids.size()
    out_size = attn_masks.size(1)
    gather_index = get_gather_index(txt_lens, num_bbs, bs, max_tl, out_size)
    dec_len = torch.stack(dec_len, dim=0)
    txt_lens = torch.tensor(txt_lens)

    max_nbb = max(num_bbs)
    ot_scatter = _compute_ot_scatter(txt_lens, max_tl, attn_masks.size(1))
    txt_pad = _compute_pad(txt_lens, max_tl)
    img_pad = _compute_pad(num_bbs, max_nbb)

    # print("len input_ids_shape: ", input_ids.shape)
    # print("len txt_lens shape: ", txt_lens.shape)
    ot_inputs = {'ot_scatter': ot_scatter,
                 'scatter_max': ot_scatter.max().item(),
                 'txt_pad': txt_pad,
                 'img_pad': img_pad,
                 'input_str':input_str,
                 'tgt_str':tgt_str}

    batch = {'qids': qids,
             'input_ids': input_ids,
             'txt_labels': txt_labels,
             'img_masks': img_mask,
             'img_mask_tgt':img_mask_tgt,
             'txt_lens':txt_lens,
             'num_bbs':num_bbs,
             'position_ids': position_ids,
             'img_feat': img_feats,
             'soft_labels': soft_labels,
             'img_pos_feat': img_pos_feat,
             'attn_masks': attn_masks,
             'gather_index': gather_index,
             "dec_batch":dec_batch,
             "dec_len":dec_len,
             'targets': targets,
             'dec_mask':dec_padding_mask,
             'dec_pos_f':dec_pos_f,
             'ot_inputs':ot_inputs}
    # print("dataset.py 392:", batch['input_ids'][0])
    return batch

def get_data_loader(args, image_dir, article_raw, tokenizer,device, image_useful_file='', 
                    image_useless_file='', random_seed=1, data_mode='test', Imode = "D+Vs",
                    balance_useful = False):

    if data_mode=='test':
        dataset = MultiDataset(args,
                                     image_dir,
                                     article_raw,
                                     tokenizer,
                                     image_useful_file,
                                     image_useless_file,
                                     random_seed,
                                     data_mode,
                                     Imode,
                                     balance_useful)
        # sampler = TokenBucketSampler(train_dataset.lens, bucket_size=BUCKET_SIZE,
        #                              batch_size=args.batch_size, droplast=False)
        sampler = None
        dataloader = DataLoader(dataset,
                                     # batch_sampler=sampler,
                                     # batch_size=4,
                                     shuffle=False,
                                     num_workers=args.n_workers,
                                     pin_memory=args.pin_mem,
                                     collate_fn=vqa_eval_collate)
        dataloader = PrefetchLoader(dataloader, device_id=device)
    else:
        dataset = MultiDataset(args,
                                     image_dir,
                                     article_raw,
                                     tokenizer,
                                     image_useful_file,
                                     image_useless_file,
                                     random_seed,
                                     data_mode,
                                     Imode,
                                     balance_useful)
        if args.distributed:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
            dataloader = DataLoader(dataset,
                                    shuffle=(sampler is None),
                                    sampler=sampler,
                                    batch_size=8,
                                    num_workers=args.n_workers,
                                    pin_memory=args.pin_mem,
                                    collate_fn=vqa_eval_collate)
        else:
            BUCKET_SIZE = 8
            sampler = TokenBucketSampler(dataset.lens, bucket_size=BUCKET_SIZE,
                                     batch_size=args.batch_size, droplast=False)
            dataloader = DataLoader(dataset,
                                    batch_sampler=sampler,
                                    num_workers=args.n_workers,
                                    pin_memory=args.pin_mem,
                                    collate_fn=vqa_eval_collate)
        dataloader = PrefetchLoader(dataloader, device_id=device)
    return dataloader, sampler



def main_tmp():
    BUCKET_SIZE = 8192

    parser = argparse.ArgumentParser()
    parser.add_argument('--n_workers', type=int, default=4,
                        help="number of data workers")
    parser.add_argument('--pin_mem', action='store_true',
                        help="pin memory")
    args = parser.parse_args()
    with open('./configs_msmo/base.json', 'r') as f:
        data = json.load(f)
        f.close()
    args.__dict__.update(data)
    tokenizer = BertTokenizer.from_pretrained(args.toker, do_lower_case='uncased' in args.toker)
    # train_dataset = MultiDataset(config=args,
    #                              images_dir=args.image_dir,
    #                              article_raw=args.article_raw,
    #                              tokenizer=tokenizer,
    #                              image_useful_file=args.train_useful_pic_path,
    #                              image_useless_file=args.train_useless_pic_path,
    #                              random_seed=1,
    #                              data_mode='test')
    # sampler = TokenBucketSampler(train_dataset.lens, bucket_size=BUCKET_SIZE,
    #                              batch_size=args.batch_size, droplast=False)
    # eval_dataloader = DataLoader(train_dataset,
    #                              batch_sampler=sampler,
    #                              # batch_size=args.batch_size,
    #                              num_workers=args.n_workers,
    #                              pin_memory=args.pin_mem,
    #                              collate_fn=vqa_eval_collate)
    device = device = torch.device("cuda", 0)
    train_dataloader, sampler = get_data_loader(args=args,
                                             image_dir=args.image_dir,
                                             article_raw=args.article_raw,
                                             tokenizer=tokenizer,
                                             device= device,
                                             image_useful_file=args.train_useful_pic_path,
                                             image_useless_file=args.train_useless_pic_path,
                                             random_seed=1,
                                             data_mode='test'
                                             )
    for idb, batch in enumerate(train_dataloader):
        print("*******{}*******:size{}".format(idb,len(batch['input_ids'])))
        if idb>10:
            break
        # import pdb
        # pdb.set_trace()
        print(batch['input_ids'])


if __name__ == '__main__':
    main_tmp()
