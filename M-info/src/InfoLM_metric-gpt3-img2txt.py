import argparse
import json
from torch.utils.data import DataLoader
import torch
from utils.misc import Struct

import sys
from time import time
import os
import logging
import glob
import pickle
from pytorch_pretrained_bert import BertTokenizer
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from tqdm import tqdm
sys.path.append('../')
from src.datasets.dataset_msmo_gpt3 import get_data_loader
from src.datasets.loader import PrefetchLoader
from src.model.pretrain_multiImg2txt import UniterForPretraining
import torch.nn.functional as F
import copy
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s: - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)
# torch.backends.cudnn.enabled = True
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
torch.manual_seed(123)
# def convert_tokens_to_string(current_sub_text):
#     return " ".join(self.convert_ids_to_tokens(tokens))

def merge_sub_word(tokenizer, sentence):
    sub_words = tokenizer.convert_ids_to_tokens(sentence)
    words = []
    i= 0
    len_sub = len(sub_words)
    cur_word = ''
    while i <len_sub:
        if sub_words[i].startswith('##'):
            cur_word = cur_word+sub_words[i][2:]
        else:
            if len(cur_word)!=0:
                words.append(cur_word)
            cur_word = sub_words[i]
        i = i+1
    if len(cur_word)!=0:
        words.append(cur_word)
    return words

def infer(config_name = './configs/config.json', model_path=None, remove_empty=True):
    BUCKET_SIZE = 8192
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_workers', type=int, default=1,
                        help="number of data workers")
    parser.add_argument('--pin_mem', action='store_true',
                        help="pin memory")
    args = parser.parse_args()
    with open(config_name, 'r') as f:
        data = json.load(f)
        f.close()
    args.__dict__.update(data)
    # args.guiding = base_config.guiding_in_test
    # args.shuffle = base_config.shuffle_in_test

    device = args.device_id
    print("device id: ",device)
    tokenizer = BertTokenizer.from_pretrained(args.toker, do_lower_case='uncased' in args.toker)
    # import pdb
    # pdb.set_trace()
    # test_image_path = "{}/test_data/img_roi".format(args.image_dir)
    # test_article_dir = "{}/test_data/article".format(args.article_dir)
    # import pdb
    # pdb.set_trace()
    CR_dataloder, _= get_data_loader(args, args.image_dir, args.article_dir, tokenizer,
                                     device, args.test_useful_pic_path, args.test_useless_pic_path,
                                     data_mode='test', Imode="hyp")
    DV_dataloder, _= get_data_loader(args, args.image_dir, args.article_dir, tokenizer,
                                     device, args.test_useful_pic_path, args.test_useless_pic_path,
                                     data_mode='test', Imode="D+Vs")
    # DR_dataloder, _= get_data_loader(args, args.image_dir, args.article_dir, tokenizer,
    #                                  device, args.test_useful_pic_path, args.test_useless_pic_path,
    #                                  data_mode='test', Imode="DR")
    # test_dataloder = PrefetchLoader(test_dataloder, device_id=device)

    # beam_Search_model = MultiModal(args, tokenizer)
    ckpt_file = args.checkpoint
    checkpoint = torch.load(ckpt_file)
    pretrained_model = UniterForPretraining.from_pretrained(args.model_config, checkpoint, img_dim=2048, img_label_dim=1601)
    cls_chkt_file = "/data-yifan/Info_checkpoints/model_8_1_1713253743"
    cls_checkpoint = torch.load(cls_chkt_file)
    pretrained_model.cls.load_state_dict(cls_checkpoint['model_cls'], strict=True)
    # import pdb
    # pdb.set_trace()
    pretrained_model.to(device)
    pretrained_model.eval()
    iter_bar = tqdm(total=len(DV_dataloder.loader.dataset), desc='Iter (loss=X.XXX)')
    kl_dv2rv_list = []
    kl_dv2cr_list = []
    kl_dv2ry_list = []
    kl_dv2r_list = []

    # for i in range(2000):
    # with open('dv2cr.p','rb') as f:
    #     dv2cr_list = pickle.load(f)
    #     f.close()
    def get_kl(pretrained_model,batch, dv_dist):
        with torch.no_grad():
            dist = pretrained_model(batch)
            kl_dv2r = -(dv_dist*torch.log(dist)).sum()
            kl2 = (dist*torch.log(dist)).sum()
            kl_dv2cr = kl_dv2r + kl2
        kl_dv2cr = kl_dv2cr.detach().cpu().item()
        return kl_dv2cr

    import time
    for i, data in enumerate(zip(DV_dataloder,CR_dataloder)):
        dv_batch,cr_batch = data[0], data[1]
        if remove_empty:
            if cr_batch['ot_inputs']['tgt_str'][0].strip()=="":
                continue
            if cr_batch['input_ids'].shape[1]<=2:
                continue
        # if i%20==0:
        #     print("i: ",i)
        # print("i: ",i)
        if i>199:
            break
        # import pdb
        # pdb.set_trace()
        with torch.no_grad():
            distribution_dv = pretrained_model(dv_batch)
        kl_dv2cr = get_kl(pretrained_model, cr_batch, distribution_dv)
        kl_dv2cr_list.append(kl_dv2cr)
        print("{}  cr: {}".format(i, kl_dv2cr))
    print("kl_dv2cr list: ", kl_dv2cr_list)
    print("kl_dv2cr average: ", sum(kl_dv2cr_list)/len(kl_dv2cr_list))

def infer_paralle(config_name = './configs/config.json', model_path=None):
    BUCKET_SIZE = 8192
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_workers', type=int, default=1,
                        help="number of data workers")
    parser.add_argument('--pin_mem', action='store_true',
                        help="pin memory")
    args = parser.parse_args()
    with open(config_name, 'r') as f:
        data = json.load(f)
        f.close()
    args.__dict__.update(data)
    # args.guiding = base_config.guiding_in_test
    # args.shuffle = base_config.shuffle_in_test

    device = args.device_id
    print("device id: ",device)
    tokenizer = BertTokenizer.from_pretrained(args.toker, do_lower_case='uncased' in args.toker)
    # import pdb
    # pdb.set_trace()
    # test_image_path = "{}/test_data/img_roi".format(args.image_dir)
    # test_article_dir = "{}/test_data/article".format(args.article_dir)
    # import pdb
    # pdb.set_trace()
    CR_dataloder, _= get_data_loader(args, args.image_dir, args.article_dir, tokenizer,
                                     device, args.test_useful_pic_path, args.test_useless_pic_path,
                                     data_mode='test', Imode="CR")
    DV_dataloder, _= get_data_loader(args, args.image_dir, args.article_dir, tokenizer,
                                     device, args.test_useful_pic_path, args.test_useless_pic_path,
                                     data_mode='test', Imode="D+Vs")
    R_dataloder, _= get_data_loader(args, args.image_dir, args.article_dir, tokenizer,
                                     device, args.test_useful_pic_path, args.test_useless_pic_path,
                                     data_mode='test', Imode="R")
    RV_dataloder, _= get_data_loader(args, args.image_dir, args.article_dir, tokenizer,
                                     device, args.test_useful_pic_path, args.test_useless_pic_path,
                                     data_mode='test', Imode="R+V")
    # Rn_dataloder, _= get_data_loader(args, args.image_dir, args.article_dir, tokenizer,
    #                                  device, args.test_useful_pic_path, args.test_useless_pic_path,
    #                                  data_mode='test', Imode="Rn")
    Ry_dataloder, _= get_data_loader(args, args.image_dir, args.article_dir, tokenizer,
                                     device, args.test_useful_pic_path, args.test_useless_pic_path,
                                     data_mode='test', Imode="Ry")
    # DR_dataloder, _= get_data_loader(args, args.image_dir, args.article_dir, tokenizer,
    #                                  device, args.test_useful_pic_path, args.test_useless_pic_path,
    #                                  data_mode='test', Imode="DR")
    # test_dataloder = PrefetchLoader(test_dataloder, device_id=device)

    # beam_Search_model = MultiModal(args, tokenizer)
    ckpt_file = args.checkpoint
    checkpoint = torch.load(ckpt_file)
    pretrained_model = UniterForPretraining.from_pretrained(args.model_config, checkpoint, img_dim=2048, img_label_dim=1601)
    cls_chkt_file = "/data-yifan/Info_checkpoints/model_8_1_1713253743"
    cls_checkpoint = torch.load(cls_chkt_file)
    pretrained_model.cls.load_state_dict(cls_checkpoint['model_cls'], strict=True)
    # import pdb
    # pdb.set_trace()
    pretrained_model.to(device)
    pretrained_model.eval()
    iter_bar = tqdm(total=len(DV_dataloder.loader.dataset), desc='Iter (loss=X.XXX)')
    kl_dv2rv_list = []
    kl_dv2cr_list = []
    kl_dv2ry_list = []
    kl_dv2r_list = []

    # for i in range(2000):
    # with open('dv2cr.p','rb') as f:
    #     dv2cr_list = pickle.load(f)
    #     f.close()
    def get_kl(pretrained_model,batch, dv_dist):
        with torch.no_grad():
            dist = pretrained_model.forward_paralle(batch)
            kl_dv2r = -(dv_dist*torch.log(dist)).sum()
            kl2 = (dist*torch.log(dist)).sum()
            kl_dv2cr = kl_dv2r + kl2
        kl_dv2cr = kl_dv2cr.detach().cpu().item()
        return kl_dv2cr

    import time
    time_seq = []
    seq_lens = []
    for i, data in enumerate(zip(DV_dataloder,Ry_dataloder,RV_dataloder, CR_dataloder, R_dataloder)):
        dv_batch,ry_batch,rv_batch, cr_batch,r_batch = data[0], data[1], data[2], data[3],data[4]
        # if i%20==0:
        #     print("i: ",i)
        print("-------------i: ",i)
        # if i>30:
        #     break
        # import pdb
        # pdb.set_trace()
        start0 = time.time()
        try:
            with torch.no_grad():
                distribution_dv = pretrained_model.forward_paralle(dv_batch)
                print("input len: {}, img_len: {}".format(dv_batch['input_ids'].shape[1],dv_batch['img_feat'].shape[1]))
            start1 = time.time()
            kl_dv2ry = get_kl(pretrained_model, ry_batch, distribution_dv)
            start2 = time.time()
            kl_dv2rv = get_kl(pretrained_model, rv_batch, distribution_dv)
            start3 = time.time()
            kl_dv2cr = get_kl(pretrained_model, cr_batch, distribution_dv)
            start4 = time.time()
            kl_dv2r = get_kl(pretrained_model, r_batch, distribution_dv)
        except:
            print("error")
        start5 = time.time()
        print("T0: ", start1-start0)
        print("T1: ", start2-start1)
        print("T2: ", start3-start2)
        print("T3: ", start4-start3)
        print("T4: ", start5-start4)
        time_seq.append(start1-start0 + start4-start3)
        seq_lens.append(dv_batch['input_ids'].shape[1]+ dv_batch['img_feat'].shape[1])
        # import pdb
        # pdb.set_trace()
        kl_dv2ry_list.append(kl_dv2ry)
        kl_dv2rv_list.append(kl_dv2rv)
        kl_dv2cr_list.append(kl_dv2cr)
        kl_dv2r_list.append(kl_dv2r)
        # if kl_dv2cr <=  dv2cr_li:
        # st[i]:
        # import pdb
        # pdb.set_trace()
        print("{}: ry: {}, rv:{} cr: {}".format(i,kl_dv2ry, kl_dv2rv, kl_dv2cr))
        # print("{}: cr: {}".format(i, kl_dv2cr))
    print(sum(time_seq), sum(seq_lens), len(seq_lens))
    print("average time: ", sum(time_seq)/len(seq_lens))
    print("kl_dv2ry list: ", kl_dv2ry_list)
    print("kl_dv2rv list: ", kl_dv2rv_list)
    print("kl_dv2cr list: ", kl_dv2cr_list)
    print("kl_dv2cr list: ", kl_dv2r_list)
    print("kl_dv2ry average: ", sum(kl_dv2ry_list)/len(kl_dv2ry_list))
    print("kl_dv2rv average: ", sum(kl_dv2rv_list)/len(kl_dv2rv_list))
    print("kl_dv2cr average: ", sum(kl_dv2cr_list)/len(kl_dv2cr_list))
    print("kl_dv2cr average: ", sum(kl_dv2r_list)/len(kl_dv2r_list))
    
"""
    for i in range(2000):
        if i<=1:
            continue
        dv_batch = next(iter(DV_dataloder))
        r_batch = next(iter(R_dataloder))
        cr_batch = next(iter(CR_dataloder))
        rn_batch = next(iter(Rn_dataloder))
        ry_batch = next(iter(Ry_dataloder))
        dr_batch = next(iter(DR_dataloder))
        with torch.no_grad():
            output_dv = pretrained_model.forward_sim(dv_batch)
            import pdb
            pdb.set_trace()
            output_r = pretrained_model.forward_sim(r_batch)
            output_cr = pretrained_model.forward_sim(cr_batch)
            output_rn = pretrained_model.forward_sim(rn_batch)
            output_ry = pretrained_model.forward_sim(ry_batch)
            output_dr = pretrained_model.forward_sim(dr_batch)
        """

        
if __name__ == '__main__':
    import sys
    config_name = sys.argv[1]
    sys.argv = sys.argv[:-1]
    infer(config_name)
