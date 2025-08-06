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

def infer(config_name = './configs/config.json', model_path=None):
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
                                     device, "", "",
                                     data_mode='test', Imode="hyp")
    DV_dataloder, _= get_data_loader(args, args.image_dir, args.article_dir, tokenizer,
                                     device, "", "",
                                     data_mode='test', Imode="D+Vs")
    
    ckpt_file = args.checkpoint
    checkpoint = torch.load(ckpt_file)
    pretrained_model = UniterForPretraining.from_pretrained(args.model_config, checkpoint, img_dim=2048, img_label_dim=1601)
    cls_chkt_file = args.ckpt_path
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
    for i, data in enumerate(zip(DV_dataloder, CR_dataloder)):
        dv_batch,cr_batch = data[0], data[1]
        import pdb
        pdb.set_trace()
        # if i%20==0:
        #     print("i: ",i)
        # print("i: ",i)
        if i>199:
            break
        import pdb
        pdb.set_trace()
        start0 = time.time()
        with torch.no_grad():
            distribution_dv = pretrained_model(dv_batch)
        start1 = time.time()
        start3 = time.time()
        kl_dv2cr = get_kl(pretrained_model, cr_batch, distribution_dv)
        start4 = time.time()
        start5 = time.time()
        print("T0: ", start1-start0)
        print("T3: ", start4-start3)
        print("T4: ", start5-start4)
        kl_dv2cr_list.append(kl_dv2cr)
        # if kl_dv2cr <=  dv2cr_li:
        # st[i]:
        # import pdb
        # pdb.set_trace()
        print("{} cr: {}".format(i, kl_dv2cr))
        # print("{}: cr: {}".format(i, kl_dv2cr))
    print("kl_dv2cr list: ", kl_dv2cr_list)
    print("kl_dv2cr average: ", sum(kl_dv2cr_list)/len(kl_dv2cr_list))

def infer_paralle(config_name = './configs/config.json', model_path=None, remove_empty=True):
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
                                     device, "", "",
                                     data_mode='test', Imode="hyp")
    DV_dataloder, _= get_data_loader(args, args.image_dir, args.article_dir, tokenizer,
                                     device, "", "",
                                     data_mode='test', Imode="D+Vs")
    
    ckpt_file = args.checkpoint
    checkpoint = torch.load(ckpt_file)
    pretrained_model = UniterForPretraining.from_pretrained(args.model_config, checkpoint, img_dim=2048, img_label_dim=1601)
    cls_chkt_file = args.ckpt_path
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
    for i, data in enumerate(zip(DV_dataloder, CR_dataloder)):
        dv_batch,cr_batch = data[0], data[1]
        if remove_empty:
            if cr_batch['ot_inputs']['tgt_str'][0].strip()=="":
                continue
        # if i%20==0:
        #     print("i: ",i)
        print("-------------i: ",i)
        if i>200:
            break
        # import pdb
        # pdb.set_trace()
        start0 = time.time()
        try:
            with torch.no_grad():
                distribution_dv = pretrained_model.forward_paralle(dv_batch)
                print("input len: {}, img_len: {}".format(dv_batch['input_ids'].shape[1],dv_batch['img_feat'].shape[1]))
            start1 = time.time()
            kl_dv2cr = get_kl(pretrained_model, cr_batch, distribution_dv)
            start4 = time.time()
        except:
            print("error")
        start5 = time.time()
        print("T0: ", start5-start0)
        seq_lens.append(dv_batch['input_ids'].shape[1]+ dv_batch['img_feat'].shape[1])
        # import pdb
        # pdb.set_trace()
        kl_dv2cr_list.append(kl_dv2cr)
        time_seq.append(start4-start1)
        print("{} cr: {}".format(i,kl_dv2cr))
    print("average time: ", sum(time_seq)/len(seq_lens))
    print("M-info list: ", kl_dv2cr_list)
    print("M-info average: ", sum(kl_dv2cr_list)/len(kl_dv2cr_list))
    

        
if __name__ == '__main__':
    import sys
    config_name = sys.argv[1]
    sys.argv = sys.argv[:-1]
    infer_paralle(config_name)
