__author__ = 'licheng'

"""
This interface provides access to four datasets:
1) refclef
2) refcoco
3) refcoco+
4) refcocog
split by unc and google

The following API functions are defined:
REFER      - REFER api class
getRefIds  - get ref ids that satisfy given filter conditions.
getAnnIds  - get ann ids that satisfy given filter conditions.
getImgIds  - get image ids that satisfy given filter conditions.
getCatIds  - get category ids that satisfy given filter conditions.
loadRefs   - load refs with the specified ref ids.
loadAnns   - load anns with the specified ann ids.
loadImgs   - load images with the specified image ids.
loadCats   - load category names with the specified category ids.
getRefBox  - get ref's bounding box [x, y, w, h] given the ref_id
showRef    - show image, segmentation or box of the referred object with the ref
getMask    - get mask and area of the referred object given ref
showMask   - show mask of the referred object given ref
"""
import sys
sys.path.append('../')
import sys
import os.path as osp
import json
import pickle
import time
import itertools
import skimage.io as io
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon, Rectangle
from pprint import pprint
import numpy as np
# from external import mask
from src.datasets.convert_imgdir import load_npz
from pytorch_pretrained_bert import BertTokenizer
import torch
from torch.nn.utils.rnn import pad_sequence
from src.datasets.data import pad_tensors, get_gather_index
from toolz.sandbox import unzip
from torch.utils.data import DataLoader, Dataset
from src.datasets.loader import PrefetchLoader
from src.datasets.sampler import TokenBucketSampler
import spacy
import pickle
import random
from random import randint
import os
eng_model = spacy.load('en_core_web_sm')
# import cv2
# from skimage.measure import label, regionprops

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

class REFER(Dataset):

	def __init__(self, data_root, dataset='refcoco', splitBy='unc', tokenizer=None):
		# provide data_root folder which contains refclef, refcoco, refcoco+ and refcocog
		# also provide dataset name and splitBy information
		# e.g., dataset = 'refcoco', splitBy = 'unc'
		print('loading dataset %s into memory...' % dataset)
		self.ROOT_DIR = osp.abspath(osp.dirname(__file__))
		self.DATA_DIR = osp.join(data_root, dataset)
		self.tokenizer = tokenizer
		if dataset in ['refcoco', 'refcoco+', 'refcocog']:
			self.IMAGE_DIR = osp.join(data_root, 'images/mscoco/images/train2014')
			self.IMAGE_ROI_DIR = osp.join(data_root, 'images/mscoco/images/train2014_roi')
		elif dataset == 'refclef':
			self.IMAGE_DIR = osp.join(data_root, 'images/saiapr_tc-12')
		else:
			print('No refer dataset is called [%s]' % dataset)
			sys.exit()

		# load refs from data/dataset/refs(dataset).json
		tic = time.time()
		ref_file = osp.join(self.DATA_DIR, 'refs('+splitBy+').p')
		self.data = {}
		self.data['dataset'] = dataset
		self.data['refs'] = pickle.load(open(ref_file, 'rb'))

		# load annotations from data/dataset/instances.json
		instances_file = osp.join(self.DATA_DIR, 'instances.json')
		instances = json.load(open(instances_file, 'r'))
		self.data['images'] = instances['images']
		self.data['annotations'] = instances['annotations']
		self.data['categories'] = instances['categories']

		# create index
		self.ref_ids = self.getRefIds(split='train')
		self.lens = len(self.ref_ids)
		with open("/data-yifan/datasets/refer_data/refcoco/refs(google)2scale.p", 'rb') as f:
			self.ref2scale = pickle.load(f)
			f.close()
		with open("/data-yifan/datasets/refer_data/refcoco/refs(google)2word.p", 'rb') as f:
			self.ref2word = pickle.load(f)
			f.close()
	
		self.createIndex()
			
		self.cls_ = 101
		self.sep = 102
		self.mask = 103
		self.max_txt_len = 20
		self.max_dec_steps = 10
		self.conf_th = 0.2
		self.max_bb = 100
		self.min_bb = 10
		self.num_bb = 36
		self.init_lens_calc("/data-yifan/datasets/refer_data/images/mscoco/images/annotations/captions_train2014.json",
					  "/data-yifan/datasets/refer_data/refcoco/input_lens.p")
		print('DONE (t=%.2fs)' % (time.time()-tic))

	def init_lens_calc(self, caption_file="", lens_file=""):
		with open(caption_file, 'r') as f:
			image2caption = json.load(f)
			f.close()
			img2caps = image2caption['annotations']
			img2caps = {ex['image_id']:ex['caption'] for ex in img2caps}
		# self.captions_lens = []
		# self.img_lens = []
		if not os.path.exists(lens_file):
			self.lens = []
			self.ref2lens = {}
			for ref_id in self.ref_ids:
				ref = self.loadRefs(ref_id)[0]
				img_id = ref['image_id']
				ref['captions'] = img2caps[img_id]
				inp, inp_len, dec = self.numerate_txt(img2caps[img_id], "")
				ref['txt_len'] = inp_len

				roi_name = ref['file_name']
				img_feat_name = "_".join(roi_name.split('_')[:-1])+'.npz'
				img_feat, img_pos_feat, num_bb, soft_labels = self._get_img_feat(self.IMAGE_ROI_DIR+"/"+img_feat_name)
				img_len = num_bb
				ref['img_len'] = img_len
				self.lens.append(inp_len+img_len)
				self.ref2lens[ref_id] = inp_len+img_len

			with open(lens_file, 'wb') as f:
				pickle.dump(self.ref2lens, f)
				f.close()
		else:
			for ref_id in self.ref_ids:
				ref = self.loadRefs(ref_id)[0]
				img_id = ref['image_id']
				ref['captions'] = img2caps[img_id]
			with open(lens_file, 'rb') as f:
				self.ref2lens = pickle.load(f)
				f.close()
			self.lens = [self.ref2lens[ref_id] for ref_id in self.ref_ids]
		
	def createIndex(self):
		# create sets of mapping
		# 1)  Refs: 	 	{ref_id: ref}
		# 2)  Anns: 	 	{ann_id: ann}
		# 3)  Imgs:		 	{image_id: image}
		# 4)  Cats: 	 	{category_id: category_name}
		# 5)  Sents:     	{sent_id: sent}
		# 6)  imgToRefs: 	{image_id: refs}
		# 7)  imgToAnns: 	{image_id: anns}
		# 8)  refToAnn:  	{ref_id: ann}
		# 9)  annToRef:  	{ann_id: ref}
		# 10) catToRefs: 	{category_id: refs}
		# 11) sentToRef: 	{sent_id: ref}
		# 12) sentToTokens: {sent_id: tokens}
		print('creating index...')
		# fetch info from instances
		Anns, Imgs, Cats, imgToAnns = {}, {}, {}, {}
		for ann in self.data['annotations']:
			Anns[ann['id']] = ann
			imgToAnns[ann['image_id']] = imgToAnns.get(ann['image_id'], []) + [ann]
		for img in self.data['images']:
			Imgs[img['id']] = img
		for cat in self.data['categories']:
			Cats[cat['id']] = cat['name']

		# fetch info from refs
		Refs, imgToRefs, refToAnn, annToRef, catToRefs = {}, {}, {}, {}, {}
		Sents, sentToRef, sentToTokens = {}, {}, {}
		for ref in self.data['refs']:
			# ids
			ref_id = ref['ref_id']
			ann_id = ref['ann_id']
			category_id = ref['category_id']
			image_id = ref['image_id']

			# add mapping related to ref
			Refs[ref_id] = ref
			imgToRefs[image_id] = imgToRefs.get(image_id, []) + [ref]
			catToRefs[category_id] = catToRefs.get(category_id, []) + [ref]
			refToAnn[ref_id] = Anns[ann_id]
			annToRef[ann_id] = ref

			# add mapping of sent
			for sent in ref['sentences']:
				Sents[sent['sent_id']] = sent
				sentToRef[sent['sent_id']] = ref
				sentToTokens[sent['sent_id']] = sent['tokens']

		# create class members
		self.Refs = Refs
		self.Anns = Anns
		self.Imgs = Imgs
		self.Cats = Cats
		self.Sents = Sents
		self.imgToRefs = imgToRefs
		self.imgToAnns = imgToAnns
		self.refToAnn = refToAnn
		self.annToRef = annToRef
		self.catToRefs = catToRefs
		self.sentToRef = sentToRef
		self.sentToTokens = sentToTokens

		print('index created.')

	def getRefIds(self, image_ids=[], cat_ids=[], ref_ids=[], split=''):
		image_ids = image_ids if type(image_ids) == list else [image_ids]
		cat_ids = cat_ids if type(cat_ids) == list else [cat_ids]
		ref_ids = ref_ids if type(ref_ids) == list else [ref_ids]

		if len(image_ids)==len(cat_ids)==len(ref_ids)==len(split)==0:
			refs = self.data['refs']
		else:
			if not len(image_ids) == 0:
				refs = [self.imgToRefs[image_id] for image_id in image_ids]
			else:
				refs = self.data['refs']
			if not len(cat_ids) == 0:
				refs = [ref for ref in refs if ref['category_id'] in cat_ids]
			if not len(ref_ids) == 0:
				refs = [ref for ref in refs if ref['ref_id'] in ref_ids]
			if not len(split) == 0:
				if split in ['testA', 'testB', 'testC']:
					refs = [ref for ref in refs if split[-1] in ref['split']] # we also consider testAB, testBC, ...
				elif split in ['testAB', 'testBC', 'testAC']:
					refs = [ref for ref in refs if ref['split'] == split]  # rarely used I guess...
				elif split == 'test':
					refs = [ref for ref in refs if 'test' in ref['split']]
				elif split == 'train' or split == 'val':
					refs = [ref for ref in refs if ref['split'] == split]
				else:
					print('No such split [%s]' % split)
					sys.exit()
		ref_ids = [ref['ref_id'] for ref in refs]
		return ref_ids

	def getAnnIds(self, image_ids=[], cat_ids=[], ref_ids=[]):
		image_ids = image_ids if type(image_ids) == list else [image_ids]
		cat_ids = cat_ids if type(cat_ids) == list else [cat_ids]
		ref_ids = ref_ids if type(ref_ids) == list else [ref_ids]

		if len(image_ids) == len(cat_ids) == len(ref_ids) == 0:
			ann_ids = [ann['id'] for ann in self.data['annotations']]
		else:
			if not len(image_ids) == 0:
				lists = [self.imgToAnns[image_id] for image_id in image_ids if image_id in self.imgToAnns]  # list of [anns]
				anns = list(itertools.chain.from_iterable(lists))
			else:
				anns = self.data['annotations']
			if not len(cat_ids) == 0:
				anns = [ann for ann in anns if ann['category_id'] in cat_ids]
			ann_ids = [ann['id'] for ann in anns]
			if not len(ref_ids) == 0:
				ids = set(ann_ids).intersection(set([self.Refs[ref_id]['ann_id'] for ref_id in ref_ids]))
		return ann_ids

	def getImgIds(self, ref_ids=[]):
		ref_ids = ref_ids if type(ref_ids) == list else [ref_ids]

		if not len(ref_ids) == 0:
			image_ids = list(set([self.Refs[ref_id]['image_id'] for ref_id in ref_ids]))
		else:
			image_ids = self.Imgs.keys()
		return image_ids

	def getCatIds(self):
		return self.Cats.keys()

	def loadRefs(self, ref_ids=[]):
		if type(ref_ids) == list:
			return [self.Refs[ref_id] for ref_id in ref_ids]
		elif type(ref_ids) == int:
			return [self.Refs[ref_ids]]

	def loadAnns(self, ann_ids=[]):
		if type(ann_ids) == list:
			return [self.Anns[ann_id] for ann_id in ann_ids]
		elif type(ann_ids) == int or type(ann_ids) == unicode:
			return [self.Anns[ann_ids]]

	def loadImgs(self, image_ids=[]):
		if type(image_ids) == list:
			return [self.Imgs[image_id] for image_id in image_ids]
		elif type(image_ids) == int:
			return [self.Imgs[image_ids]]

	def loadCats(self, cat_ids=[]):
		if type(cat_ids) == list:
			return [self.Cats[cat_id] for cat_id in cat_ids]
		elif type(cat_ids) == int:
			return [self.Cats[cat_ids]]

	def getRefBox(self, ref_id):
		ref = self.Refs[ref_id]
		ann = self.refToAnn[ref_id]
		return ann['bbox']  # [x, y, w, h]

	def showRef(self, ref, seg_box='seg'):
		ax = plt.gca()
		# show image
		image = self.Imgs[ref['image_id']]
		I = io.imread(osp.join(self.IMAGE_DIR, image['file_name']))
		ax.imshow(I)
		# show refer expression
		for sid, sent in enumerate(ref['sentences']):
			print('%s. %s' % (sid+1, sent['sent']))
		# show segmentations
		if seg_box == 'seg':
			ann_id = ref['ann_id']
			ann = self.Anns[ann_id]
			polygons = []
			color = []
			c = 'none'
			if type(ann['segmentation'][0]) == list:
				# polygon used for refcoco*
				for seg in ann['segmentation']:
					poly = np.array(seg).reshape((len(seg)/2, 2))
					polygons.append(Polygon(poly, True, alpha=0.4))
					color.append(c)
				p = PatchCollection(polygons, facecolors=color, edgecolors=(1,1,0,0), linewidths=3, alpha=1)
				ax.add_collection(p)  # thick yellow polygon
				p = PatchCollection(polygons, facecolors=color, edgecolors=(1,0,0,0), linewidths=1, alpha=1)
				ax.add_collection(p)  # thin red polygon
			else:
				# mask used for refclef
				rle = ann['segmentation']
				m = mask.decode(rle)
				img = np.ones( (m.shape[0], m.shape[1], 3) )
				color_mask = np.array([2.0,166.0,101.0])/255
				for i in range(3):
					img[:,:,i] = color_mask[i]
				ax.imshow(np.dstack( (img, m*0.5) ))
		# show bounding-box
		elif seg_box == 'box':
			ann_id = ref['ann_id']
			ann = self.Anns[ann_id]
			bbox = 	self.getRefBox(ref['ref_id'])
			print("bbox: ", bbox)
			box_plot = Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], fill=False, edgecolor='green', linewidth=3)
			ax.add_patch(box_plot)

	def getMask(self, ref):
		# return mask, area and mask-center
		ann = self.refToAnn[ref['ref_id']]
		image = self.Imgs[ref['image_id']]
		if type(ann['segmentation'][0]) == list: # polygon
			rle = mask.frPyObjects(ann['segmentation'], image['height'], image['width'])
		else:
			rle = ann['segmentation']
		m = mask.decode(rle)
		m = np.sum(m, axis=2)  # sometimes there are multiple binary map (corresponding to multiple segs)
		m = m.astype(np.uint8) # convert to np.uint8
		# compute area
		area = sum(mask.area(rle))  # should be close to ann['area']
		return {'mask': m, 'area': area}
		# # position
		# position_x = np.mean(np.where(m==1)[1]) # [1] means columns (matlab style) -> x (c style)
		# position_y = np.mean(np.where(m==1)[0]) # [0] means rows (matlab style)    -> y (c style)
		# # mass position (if there were multiple regions, we use the largest one.)
		# label_m = label(m, connectivity=m.ndim)
		# regions = regionprops(label_m)
		# if len(regions) > 0:
		# 	largest_id = np.argmax(np.array([props.filled_area for props in regions]))
		# 	largest_props = regions[largest_id]
		# 	mass_y, mass_x = largest_props.centroid
		# else:
		# 	mass_x, mass_y = position_x, position_y
		# # if centroid is not in mask, we find the closest point to it from mask
		# if m[mass_y, mass_x] != 1:
		# 	print 'Finding closes mask point ...'
		# 	kernel = np.ones((10, 10),np.uint8)
		# 	me = cv2.erode(m, kernel, iterations = 1)
		# 	points = zip(np.where(me == 1)[0].tolist(), np.where(me == 1)[1].tolist())  # row, col style
		# 	points = np.array(points)
		# 	dist   = np.sum((points - (mass_y, mass_x))**2, axis=1)
		# 	id     = np.argsort(dist)[0]
		# 	mass_y, mass_x = points[id]
		# 	# return
		# return {'mask': m, 'area': area, 'position_x': position_x, 'position_y': position_y, 'mass_x': mass_x, 'mass_y': mass_y}
		# # show image and mask
		# I = io.imread(osp.join(self.IMAGE_DIR, image['file_name']))
		# plt.figure()
		# plt.imshow(I)
		# ax = plt.gca()
		# img = np.ones( (m.shape[0], m.shape[1], 3) )
		# color_mask = np.array([2.0,166.0,101.0])/255
		# for i in range(3):
		#     img[:,:,i] = color_mask[i]
		# ax.imshow(np.dstack( (img, m*0.5) ))
		# plt.show()

	def showMask(self, ref):
		M = self.getMask(ref)
		msk = M['mask']
		ax = plt.gca()
		ax.imshow(msk)

	def _get_img_feat(self, filename):
		name, dump, nbb = load_npz(self.conf_th,
                                   self.max_bb,
                                   self.min_bb,
                                   self.num_bb,
                                   filename)
		img_feat = dump['features']
		img_bb = dump['norm_bb']
		soft_labels = dump['soft_labels']
		
		img_feat = torch.tensor(img_feat[:nbb, :]).float()
		img_bb = torch.tensor(img_bb[:nbb, :]).float()
		
		img_bb = torch.cat([img_bb, img_bb[:, 4:5] * img_bb[:, 5:]], dim=-1)
		
		return img_feat, img_bb, nbb, soft_labels

	def _get_txt_feat(self,inp,dec):
		input_ids = torch.tensor(inp)
		# input_poses = self.input_poses[index]
		_dec_id = dec

		dec_inp, dec_tgt = self.get_dec_inp_targ_seqs(_dec_id,
														self.max_dec_steps,
														self.cls_,
														self.sep)
		dec_len = len(dec_inp)
		dec_inp, dec_tgt = self.pad_decoder_inp_targ(self.max_dec_steps,
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

	def numerate_txt(self, title, summary):
		titles_pre = title.split(' ')
		titles_pre = titles_pre[:self.max_txt_len]
		titles_pre = " ".join(titles_pre)
		inp = self.bert_tokenize(titles_pre.strip())
		if len(inp) > self.max_txt_len:
			inp = [self.cls_] + inp[:self.max_txt_len] + [self.sep]
		else:
			inp = [self.cls_] + inp + [self.sep]
		dec = self.bert_tokenize(summary.strip())
		# if self.config.key_w_loss:
		#     dec_pos, org_pos, dec_to_tokens = self.get_inp_pos(dec)
		# else:
		#     dec_pos = [0]
		return inp, len(inp), dec
	
	def bert_tokenize(self, text):
		ids = []
		for word in text.strip().split():
			ws = self.tokenizer.tokenize(word)
			if not ws:
				# some special char
				continue
			ids.extend(self.tokenizer.convert_tokens_to_ids(ws))
		return ids
	
	def __len__(self):
		return len(self.ref_ids)

	def __getitem__(self, index):
		ref_id = self.ref_ids[index]
		ref = self.loadRefs(ref_id)[0]
		roi_name = ref['file_name']
		img_feat_name = "_".join(roi_name.split('_')[:-1])+'.npz'
		src_text, summary = ref['captions'], random_ref2word(self.ref2word[ref_id], 2)
		
        # txt_pickle_path = self.images_dir + '/{}/article_{}/{}.pickle'.format(dn, CNAME, txt_name)
		inp, inp_len, dec = self.numerate_txt(src_text, summary)
		# inp_to_tokens, sub_words = self.merge_sub_word(inp)
		# phrase_tensor = structure_parse.to_phrase(nlp, inp_to_tokens, sub_words)
		input_ids, dec_batch, target_batch, dec_padding_mask, dec_len, copy_position = self._get_txt_feat(inp, dec)

		# img_list = glob.glob(self.images_dir + '/{}/img_roi/{}*.npz'.format(dn, ex_name))
		img_feat, img_pos_feat, num_bb, soft_labels = self._get_img_feat(self.IMAGE_ROI_DIR+"/"+img_feat_name)
		
		input_lens = inp_len + num_bb
		attn_masks = torch.ones(input_lens, dtype=torch.long)
		soft_labels = torch.tensor(soft_labels)
		
		txt_labels = torch.zeros(input_lens)-1
		# 1. mask image region
		img_w, img_h = self.ref2scale[ref_id]
		img_bboxes = img_pos_feat[:,:4].clone()
		img_bboxes[:,0] = img_bboxes[:,0] * img_w
		img_bboxes[:,1] = img_bboxes[:,1] * img_h
		img_bboxes[:,2] = img_bboxes[:,2] * img_w
		img_bboxes[:,3] = img_bboxes[:,3] * img_h

		bbox = 	self.getRefBox(ref_id)
		bbox[2] = bbox[2] + bbox[0]
		bbox[3] = bbox[3] + bbox[1]
		mask_region_id = match_bbox(bbox, img_bboxes)

		#2. mask txt 
		rand_pos, tgt_mask = random_mask(input_ids, inp_len, self.mask)

		# 3. construct label (include txt and img)
		txt_labels[inp_len + mask_region_id] = dec[0]
		txt_labels[rand_pos] = tgt_mask
		txt_labels = txt_labels.long()
		attn_masks[inp_len + mask_region_id]=0
		attn_masks[rand_pos]=0

		
		return index, input_lens, input_ids, txt_labels,\
               img_feat, soft_labels, img_pos_feat, attn_masks, \
               dec_batch, target_batch, dec_padding_mask, dec_len, torch.tensor(mask_region_id)

def random_mask(tokens, inp_len, mask):
	rand_pos = randint(1,inp_len-2)
	tgt_mask = tokens[rand_pos].clone()
	tokens[rand_pos] = mask
	return rand_pos, tgt_mask

def random_ref2word(cand_words, prior_num = 2):
	rand_scale = min(len(cand_words), prior_num)
	rand_pos = randint(0,rand_scale-1)
	# print("rand_pos: ",rand_pos)
	# print("rand_scale: ",rand_scale)
	return cand_words[rand_pos]


def vqa_eval_collate(inputs):
	(qids, input_lens, input_ids, txt_labels,
		img_feats, soft_labels, img_pos_feats, attn_masks,
		dec_batch, target_batch, dec_padding_mask,dec_len, mask_region_id) = map(list, unzip(inputs))
	txt_lens = [i.size(0) for i in input_ids]
	# print("len qids: ", len(qids))
	# print("len input_ids_lens: ", len(input_ids))
	# print("len txt_lens: ", len(txt_lens))

	input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
	# copy_position = pad_sequence(copy_position, batch_first=True, padding_value=0)
	txt_labels = pad_sequence(txt_labels, batch_first=True, padding_value=-1)
	# img_mask = pad_sequence(img_mask, batch_first=True, padding_value=0)
	# img_mask_tgt = pad_sequence(img_mask_tgt, batch_first=True, padding_value=0)
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
	mask_region_id = torch.stack(mask_region_id, dim=0)
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
					'img_pad': img_pad}

	batch = {'qids': qids,
				'input_ids': input_ids,
				'txt_labels': txt_labels,
				# 'img_masks': img_mask,
				# 'img_mask_tgt':img_mask_tgt,
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
				'mask_region_id':mask_region_id,
				'ot_inputs':ot_inputs}
	# print("dataset.py 392:", batch['input_ids'][0])
	return batch

def get_data_loader(device, tokenizer, bucket_size, data_mode='test'):
	
	# tokenizer = BertTokenizer.from_pretrained("bert-base-cased", do_lower_case='uncased' in "bert-base-cased")
	if data_mode=='test':
		dataset = REFER(data_root = "/data-yifan/datasets/refer_data", dataset='refcoco', splitBy='google', tokenizer=tokenizer)
		# sampler = TokenBucketSampler(train_dataset.lens, bucket_size=BUCKET_SIZE,
		#                              batch_size=args.batch_size, droplast=False)
		sampler = None
		dataloader = DataLoader(dataset,
										# batch_sampler=sampler,
										# batch_size=4,
										shuffle=False,
										num_workers=1,
										pin_memory=False,
										collate_fn=vqa_eval_collate)
		dataloader = PrefetchLoader(dataloader, device_id=device)
	else:
		dataset = REFER(data_root = "/data-yifan/datasets/refer_data", dataset='refcoco', splitBy='google', tokenizer=tokenizer)
		# BUCKET_SIZE = 8
		sampler = TokenBucketSampler(dataset.lens, bucket_size=bucket_size,
									batch_size=4096, droplast=False)
		dataloader = DataLoader(dataset,
								batch_sampler=sampler,
								num_workers=1,
								pin_memory=False,
								collate_fn=vqa_eval_collate)
		dataloader = PrefetchLoader(dataloader, device_id=device)
	return dataloader, sampler

def cal_iou_xyxy(box1,box2):
    x1min, y1min, x1max, y1max = box1[0], box1[1], box1[2], box1[3]
    x2min, y2min, x2max, y2max = box2[0], box2[1], box2[2], box2[3]
    #计算两个框的面积
    s1 = (y1max - y1min + 1.) * (x1max - x1min + 1.)
    s2 = (y2max - y2min + 1.) * (x2max - x2min + 1.)

    #计算相交部分的坐标
    xmin = max(x1min,x2min)
    ymin = max(y1min,y2min)
    xmax = min(x1max,x2max)
    ymax = min(y1max,y2max)

    inter_h = max(ymax - ymin + 1, 0)
    inter_w = max(xmax - xmin + 1, 0)

    intersection = inter_h * inter_w
    union = s1 + s2 - intersection

    #计算iou
    iou = intersection / union
    return iou

def match_bbox(label_bbox, auto_bboxes):
	max_iou = -1
	max_iou_id = -1
	for i, abb in enumerate(auto_bboxes):
		iou = cal_iou_xyxy(label_bbox,abb)
		if iou > max_iou:
			max_iou = iou
			max_iou_id = i
		# print("i:{} iou: {}".format(i, iou))
	return max_iou_id

def extract_mask_word(ref, cand_num = 3):
	# ref = {'ann_id': 466635,
	# 'category_id': 1,
	# 'file_name': 'COCO_train2014_000000579440_0.jpg',
	# 'image_id': 579440,
	# 'ref_id': 216,
	# 'sent_ids': [618, 619, 620],
	# 'sentences': [{'raw': 'person in chair at left',
	# 				'sent': 'person in chair at left',
	# 				'sent_id': 618,
	# 				'tokens': ['person', 'in', 'chair', 'at', 'left']},
	# 			{'raw': 'guy sitting on left',
	# 				'sent': 'guy sitting on left',
	# 				'sent_id': 619,
	# 				'tokens': ['guy', 'sitting', 'on', 'left']},
	# 			{'raw': 'black shirt in chair on left',
	# 				'sent': 'black shirt in chair on left',
	# 				'sent_id': 620,
	# 				'tokens': ['black', 'shirt', 'in', 'chair', 'on', 'left']}],
	# 'split': 'train'}
	# ref={'sent_ids': [28440, 28441, 28442], 'file_name': 'COCO_train2014_000000466532_0.jpg', 'ann_id': 1042662, 'ref_id': 9990, 'image_id': 466532, 'split': 'train', 'sentences': [{'tokens': ['green'], 'raw': 'Green', 'sent_id': 28440, 'sent': 'green'}, {'tokens': ['green', 'bananas'], 'raw': 'green bananas', 'sent_id': 28441, 'sent': 'green bananas'}, {'tokens': ['frontmiddle', 'bananas'], 'raw': 'front/middle bananas', 'sent_id': 28442, 'sent': 'frontmiddle bananas'}], 'category_id': 52}
	sents = [ex['sent'] for ex in ref['sentences']]
	cands = {}
	KEY_POSES = {'NOUN':1, 'ADJ':1, "VERB":0.8,"PROPN":0.8, 'ADJ':0.5,
			  "ADV":0.1, 'PRON':0.1, 'NUM':0.1, "PART":0.1,}
	# import pdb
	# pdb.set_trace()
	for sent in sents:
		res_pos = eng_model(sent)
		words = [str(w) for w in res_pos]
		words_poses = [w.pos_ for w in res_pos]
		for w,wp in zip(words, words_poses):
			if w=='left' or w=='right':
				continue
			if wp in KEY_POSES:
				if w in cands:
					cands[w] = cands[w] + KEY_POSES[wp]
				else:
					cands[w] = KEY_POSES[wp]
			else:
				if w in cands:
					cands[w] = cands[w] + 0.01
				else:
					cands[w] = 0.01
	cands = sorted(cands.items(),key=lambda x:x[1], reverse=True)
	cands = cands[:cand_num]
	cand_words = [x[0] for x in cands]
	# if len(cands)==0:
	return sents, cands, cand_words

def batch_extract_mask_word():
	tokenizer = None
	refer = REFER(data_root = "/data-yifan/datasets/refer_data", dataset='refcoco', splitBy='google', tokenizer=tokenizer)
	ref_ids = refer.getRefIds()
	ref2word = {}
	for i,ref_id in enumerate(ref_ids):
		ref = refer.loadRefs(ref_id)[0]
		sents, cands, mask_word = extract_mask_word(ref)
		ref2word[ref_id] = mask_word
		# if i<1300:
		# 	continue
		if i%100==0:
			print(i)
			print("sents: ", sents)
			print("cands: ", cands)
			print("mask_word: ", mask_word)
	with open("/data-yifan/datasets/refer_data/refcoco/refs(google)2word.p", 'wb') as f:
		pickle.dump(ref2word, f)
		f.close()

def batch_image_scale():
	from PIL import Image
	tokenizer = None
	refer = REFER(data_root = "/data-yifan/datasets/refer_data", dataset='refcoco', splitBy='google', tokenizer=tokenizer)
	ref_ids = refer.getRefIds()
	image_jpg_dir = "/data-yifan/datasets/refer_data/images/mscoco/images/train2014"
	ref2scale = {}

	for i,ref_id in enumerate(ref_ids):
		ref = refer.loadRefs(ref_id)[0]
		roi_name = ref['file_name']
		img_file_name = "_".join(roi_name.split('_')[:-1])+'.jpg'
		img = Image.open(image_jpg_dir + "/" + img_file_name)
		w, h = img.size
		ref2scale[ref_id] = (w, h)
		if i%100==0:
			print(i)
	with open("/data-yifan/datasets/refer_data/refcoco/refs(google)2scale.p", 'wb') as f:
		pickle.dump(ref2scale, f)
		f.close()

if __name__ == '__main__':
	"""
	# 1. 首先获取每个图片对应的单词
	# 计算该函数时，REFER没有父类。（第三步作为数据集时，REFER父类为dataset)
	# extract_mask_word('')
	batch_extract_mask_word()
	exit(0)
	"""
	"""
	# 2. 首先获取每张图片对应的长宽（因为uniter图片特征提取器，获得的是归一化的目标坐标）
	# 计算该函数时，REFER没有父类。（第三步作为数据集时，REFER父类为dataset)
	batch_image_scale()
	exit(0)
	"""

	tokenizer = BertTokenizer.from_pretrained("bert-base-cased", do_lower_case='uncased' in "bert-base-cased")
	refer = REFER(data_root = "/data-yifan/datasets/refer_data", dataset='refcoco', splitBy='google', tokenizer=tokenizer)
	ref_ids = refer.getRefIds()
	print(len(ref_ids))

	print(len(refer.Imgs))
	print(len(refer.imgToRefs))

	ref_ids = refer.getRefIds(split='train')
	
	print('There are %s training referred objects.' % len(ref_ids))
	import pdb
	pdb.set_trace()
	batch = refer.__getitem__(4)
	print("batch: ")
	print(batch)
	exit(0)
	for i,ref_id in enumerate(ref_ids):
		ref = refer.loadRefs(ref_id)[0]
		if len(ref['sentences']) < 2:
			continue
		print("i: ", i)
		pprint(ref['file_name'])

		print('The label is %s.' % refer.Cats[ref['category_id']])
		plt.figure()
		refer.showRef(ref, seg_box='box')
		plt.show()
		if i>=30:
			break
		# plt.figure()
		# refer.showMask(ref)
		# plt.show()