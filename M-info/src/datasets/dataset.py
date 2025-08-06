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
    # import pdb
    # pdb.set_trace()
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

class MultiDataset(Dataset):
    def __init__(self, config, images_path, titles_file,sent_summaris_file, tokenizer, image_useful_file, image_useless_file, random_seed, is_test=False, balance_useful= False):
        self.config = config
        self.images_dir = images_path

        self.titles=open(titles_file,'r').readlines()
        self.tokenizer =  tokenizer
        self.sent_summarizations=open(sent_summaris_file,'r').readlines()

        if config.comple_of_high_freq:
            image_useful_file = image_useless_file[:-7]+'_comple.pickle'
        self.image_useful =  pickle.load(open(image_useful_file, 'rb'))
        self.image_useless = pickle.load(open(image_useless_file, 'rb'))
        if self.config.remove_high_freq:
            self.image_useful = set(self.image_useful)-self.image_useless
        # Init and Build vocab
        self.random_seed = random_seed
        self.balance_useful = balance_useful


        # if not is_test:
        #     random.seed(random_seed)
        #     all_examples = [(self.titles[i], self.sent_summarizations[i], self.imgs[i]) for i in range(len(self.titles))]
        #     random.shuffle(all_examples)
        #     self.titles = [ex[0] for ex in all_examples]
        #     self.sent_summarizations = [ex[1] for ex in all_examples]
        #     self.imgs = [ex[2] for ex in all_examples]

        self.start_num = 0

        meta = json.load(open(config.meta_file, 'r'))
        self.cls_ = meta['CLS']
        self.sep = meta['SEP']
        self.mask = meta['MASK']
        self.v_range = meta['v_range']
        self.input_ids, self.txtlens, self.dec_ids, self.dec_poses = \
            self._get_ids_and_lens(self.titles, self.sent_summarizations)

        name2nbb_name = "/".join(images_path.split('/')[:-1])+'/'+images_path.split('/')[-1].split('_')[1]+"_name2nbb.pkl"
        if not os.path.exists(name2nbb_name):
            self.name2nbb = self._get_name_to_nbb(self.images_dir)
            with open(name2nbb_name, 'wb') as fnb:
                pickle.dump(self.name2nbb, fnb)
                fnb.close()
        else:
            with open(name2nbb_name, 'rb') as fnb:
                self.name2nbb = pickle.load(fnb)
                fnb.close()

        assert len(self.txtlens) == len(self.name2nbb)
        if self.balance_useful:
            self.sample_image_proj = sample_balance_useless(len(self.name2nbb), self.image_useful, random_seed)
        else:
            self.sample_image_proj = None

        if self.config.shuffle:
            self.shuffle_image_proj = shuffle_image(len(self.name2nbb), self.image_useful, self.random_seed)
            if self.balance_useful:
                self.lens = []
                for id in range(len(self.sample_image_proj)):
                    proj_id = self.sample_image_proj[id]
                    shuffle_id = self.shuffle_image_proj[str(proj_id + 1)]
                    self.lens.append(self.txtlens[proj_id] + self.name2nbb[shuffle_id+ '.npz'])
            else:
                self.lens = [tl + self.name2nbb[self.shuffle_image_proj[str(id + 1)] + '.npz'] for id, tl in
                             enumerate(self.txtlens)]
        else:
            if self.balance_useful:
                self.lens = []
                for id in range(len(self.sample_image_proj)):
                    proj_id = self.sample_image_proj[id]
                    self.lens.append(self.txtlens[proj_id] + self.name2nbb[str(proj_id + 1) + '.npz'])
            else:
                self.lens = [tl + self.name2nbb[str(id + 1) + '.npz'] for id, tl in enumerate(self.txtlens)]


    def __len__(self):
        return len(self.lens)

    def __getitem__(self, index: int):
        # print("dataset.py 160 index: ",index)

        if self.config.textonly:
            input_lens = torch.tensor(self.lens[index])
            img_feat, img_pos_feat, num_bb, soft_labels = self._get_img_feat(self.config.avg_img_npz)
            img_useful = 0
            input_ids, dec_batch, target_batch, dec_padding_mask, dec_len = self._get_txt_feat(index)
        else:
            if self.config.shuffle:
                input_lens = torch.tensor(self.lens[index])
                if self.balance_useful:
                    index = self.sample_image_proj[index]
                img_index = self.shuffle_image_proj[str(index + 1)]
                img_feat, img_pos_feat, num_bb, soft_labels = self._get_img_feat(self.images_dir + '/{}.npz'.format(img_index))
                img_useful = 1 if img_index + '.jpg' in self.image_useful else 0
                input_ids, dec_batch, target_batch, dec_padding_mask, dec_len = self._get_txt_feat(index)
            elif self.config.avg_img:
                input_lens = torch.tensor(self.lens[index])
                if self.balance_useful:
                    index = self.sample_image_proj[index]
                img_index = str(index + 1)
                img_useful = 1 if img_index + '.jpg' in self.image_useful else 0
                if img_useful == 0:
                    img_feat, img_pos_feat, num_bb, soft_labels = self._get_img_feat(self.config.avg_img_npz)
                else:
                    img_feat, img_pos_feat, num_bb, soft_labels = self._get_img_feat(self.images_dir + '/{}.npz'.format(img_index))
                input_ids, dec_batch, target_batch, dec_padding_mask, dec_len = self._get_txt_feat(index)
            else: #self.config.guiding
                input_lens = torch.tensor(self.lens[index])
                if self.balance_useful:
                    index = self.sample_image_proj[index]
                img_index = str(index + 1)
                img_feat, img_pos_feat, num_bb, soft_labels = self._get_img_feat(self.images_dir + '/{}.npz'.format(img_index))
                img_useful = 1 if img_index + '.jpg' in self.image_useful else 0
                input_ids, dec_batch, target_batch, dec_padding_mask, dec_len = self._get_txt_feat(index)


        if self.config.key_w_loss:
            dec_pos = [self.config.key_w_pos[0]] + self.dec_poses[index]
            dec_pos_f = [
                self.config.key_loss_weight[0] if p in self.config.key_w_pos else self.config.key_loss_weight[1]
                for p in dec_pos]
            dec_pos_f = torch.tensor(dec_pos_f)
            if len(dec_pos_f) > dec_len:
                dec_pos_f = dec_pos_f[:dec_len]
            assert len(dec_pos_f) == dec_len
        else:
            dec_pos_f = torch.tensor([0.0])

        attn_masks = torch.ones(len(input_ids) + num_bb, dtype=torch.long)
        soft_labels = torch.tensor(soft_labels)

        return index, input_lens, input_ids, img_feat, soft_labels, img_useful, img_pos_feat, attn_masks, dec_batch, target_batch, dec_padding_mask, dec_len, dec_pos_f

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

    def _get_name_to_nbb(self, image_dir):
        name2nbb = {}
        pts = glob.glob(image_dir+'/*.npz')
        for pt in pts:
            name, dump, nbb = load_npz(self.config.conf_th,
                                       self.config.max_bb,
                                       self.config.min_bb,
                                       self.config.num_bb,
                                       pt)
            name = pt.split('/')[-1]
            name2nbb[name] = nbb

        return name2nbb

    def _get_ids_and_lens(self, titles, summaris):
        assert len(titles)==len(summaris)
        lens = []
        input_ids = []
        dec_poses = []
        dec_ids = []
        for ind in range(len(titles)):
            inp = self.bert_tokenize(titles[ind].strip())
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
                dec_pos=[0]
            dec_ids.append(dec)
            dec_poses.append(dec_pos)

        return input_ids, lens, dec_ids, dec_poses

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

    def _get_txt_feat(self,index):
        input_ids = torch.tensor(self.input_ids[index])
        _dec_id = self.dec_ids[index]

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
        return input_ids, dec_inp, dec_tgt, dec_padding_mask, dec_len


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
        (qids, input_lens, input_ids, img_feats, soft_labels, img_useful, img_pos_feats, attn_masks, dec_batch, target_batch, dec_padding_mask,
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

    (qids, input_lens, input_ids, img_feats, soft_labels, img_useful, img_pos_feats, attn_masks, dec_batch, target_batch, dec_padding_mask,
     dec_len, dec_pos_f
     ) = map(list, unzip(inputs))

    txt_lens = [i.size(0) for i in input_ids]

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long
                                ).unsqueeze(0)
    attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0)
    # if targets[0] is None:
    #     targets = None
    # else:
    #     targets = torch.stack(targets, dim=0)
    dec_batch = pad_sequence(dec_batch, batch_first=True, padding_value=0)
    targets = pad_sequence(target_batch, batch_first=True, padding_value=0)
    dec_padding_mask = pad_sequence(dec_padding_mask, batch_first=True, padding_value=0)
    dec_pos_f = pad_sequence(dec_pos_f, batch_first=True, padding_value=0)
    soft_labels = pad_sequence(soft_labels, batch_first=True, padding_value=0)
    soft_labels = soft_labels.float()

    num_bbs = [f.size(0) for f in img_feats]
    img_feat = pad_tensors(img_feats, num_bbs)
    img_pos_feat = pad_tensors(img_pos_feats, num_bbs)

    bs, max_tl = input_ids.size()
    out_size = attn_masks.size(1)
    gather_index = get_gather_index(txt_lens, num_bbs, bs, max_tl, out_size)
    dec_len = torch.stack(dec_len, dim=0)
    img_useful = [base_config.useful_weight if im_u ==1  else base_config.useless_weight for im_u in img_useful]
    img_useful = torch.tensor(img_useful).unsqueeze(1).unsqueeze(1)

    max_nbb = max(num_bbs)
    ot_scatter = _compute_ot_scatter(txt_lens, max_tl, attn_masks.size(1))
    txt_pad = _compute_pad(txt_lens, max_tl)
    img_pad = _compute_pad(num_bbs, max_nbb)
    ot_inputs = {'ot_scatter': ot_scatter,
                 'scatter_max': ot_scatter.max().item(),
                 'txt_pad': txt_pad,
                 'img_pad': img_pad}

    batch = {'qids': qids,
             'input_ids': input_ids,
             'txt_lens':txt_lens,
             'position_ids': position_ids,
             'img_feat': img_feat,
             'soft_labels': soft_labels,
             'img_useful': img_useful,
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

def get_data_loader(args, dev_image_path, dev_text_path, dev_summri_path, tokenizer,device, image_useful_file='', image_useless_file='', random_seed=1, is_test=False, balance_useful = False):
    BUCKET_SIZE = 8
    if is_test:
        train_dataset = MultiDataset(args,
                                     dev_image_path,
                                     dev_text_path,
                                     dev_summri_path,
                                     tokenizer,
                                     image_useful_file,
                                     image_useless_file,
                                     random_seed,
                                     is_test,
                                     balance_useful)
        # sampler = TokenBucketSampler(train_dataset.lens, bucket_size=BUCKET_SIZE,
        #                              batch_size=args.batch_size, droplast=False)
        eval_dataloader = DataLoader(train_dataset,
                                     # batch_sampler=sampler,
                                     # batch_size=4,
                                     shuffle=False,
                                     num_workers=args.n_workers,
                                     pin_memory=args.pin_mem,
                                     collate_fn=vqa_eval_collate)
        eval_dataloader = PrefetchLoader(eval_dataloader, device_id=device)
    else:
        train_dataset = MultiDataset(args,
                                     dev_image_path,
                                     dev_text_path,
                                     dev_summri_path,
                                     tokenizer,
                                     image_useful_file,
                                     image_useless_file,
                                     random_seed,
                                     is_test,
                                     balance_useful)
        sampler = TokenBucketSampler(train_dataset.lens, bucket_size=BUCKET_SIZE,
                                     batch_size=args.batch_size, droplast=False)
        eval_dataloader = DataLoader(train_dataset,
                                     batch_sampler=sampler,
                                     # batch_size=4,
                                     num_workers=args.n_workers,
                                     pin_memory=args.pin_mem,
                                     collate_fn=vqa_eval_collate)
        eval_dataloader = PrefetchLoader(eval_dataloader, device_id=device)
    return eval_dataloader



def main_tmp():
    from sampler import TokenBucketSampler
    BUCKET_SIZE = 8192

    parser = argparse.ArgumentParser()
    parser.add_argument('--n_workers', type=int, default=4,
                        help="number of data workers")
    parser.add_argument('--pin_mem', action='store_true',
                        help="pin memory")
    args = parser.parse_args()
    with open('./configs/config.json', 'r') as f:
        data = json.load(f)
        f.close()
    args.__dict__.update(data)
    train_dataset = MultiDataset(args, args.dev_image_path,
                                 args.dev_text_path,
                                 args.dev_summri_path,
                                 'image_useful_file',
                                 'image_useless_file',
                                 1,
                                 is_test=False)
    sampler = TokenBucketSampler(train_dataset.lens, bucket_size=BUCKET_SIZE,
                                 batch_size=args.batch_size, droplast=False)
    eval_dataloader = DataLoader(train_dataset,
                                 batch_sampler=sampler,
                                 # batch_size=args.batch_size,
                                 num_workers=args.n_workers,
                                 pin_memory=args.pin_mem,
                                 collate_fn=vqa_eval_collate)
    for idb, batch in enumerate(eval_dataloader):
        print("*******{}*******".format(idb))
        import pdb
        pdb.set_trace()
        print(batch['input_ids'])


if __name__ == '__main__':
    main_tmp()
