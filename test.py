import os
import re
import copy
import json
import torch
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from model import SegModel
from eval_utils import evaluate_dataset
from transformers import BertTokenizer, set_seed


DATASET = {'doc':'doc2dial', '711':'dialseg711', 'vn_val':'vn_synth_val', 'vn_test':'vn_synth_test'}

def infer(args, model_path):
	tokenizer = BertTokenizer.from_pretrained(args.coheren_model_name)

	model = SegModel(topic_model_name=args.topic_model_name, coheren_model_name=args.coheren_model_name)
	model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')), False)
	model.to(args.device)
	model.eval()

	path_input_docs = f'./data/{args.dataset}'
	pk, wd = evaluate_dataset(model, tokenizer, path_input_docs, args.device, args.max_len,
	                           window_size=args.window_size,
	                           oracle_boundary_count=args.oracle_boundary_count,
	                           pick_num=args.pick_num)

	print('pk: ', pk)
	print('wd: ', wd)
	res = str(round(pk, 4)) + "\t" + str(round(wd, 4))
	print(f'Saving result to {args.root}/metric/{args.model}/{args.save_name}.json')
	json.dump(res, open(f'{args.root}/metric/{args.model}/{args.save_name}.json', 'w'))

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--model", required=True)
	parser.add_argument("--dataset", required=True)
	parser.add_argument("--save_name", default='epoch')

	parser.add_argument("--single_ckpt", action='store_true')
	parser.add_argument("--ckpt")
	parser.add_argument("--root", default='.')	
	parser.add_argument("--no_cuda", action='store_true')
	parser.add_argument("--ckpt_start", type=int, default=0)
	parser.add_argument("--ckpt_end", type=int, default=3)
	parser.add_argument("--pick_num", type=int, default=4)
	parser.add_argument("--oracle_boundary_count", action='store_true',
	                     help='Use the true number of boundaries per document instead of a fixed --pick_num (needed when boundary count varies, e.g. vn_synth)')
	parser.add_argument("--window_size", default=2, type=int)
	parser.add_argument("--topic_model_name", default='princeton-nlp/sup-simcse-bert-base-uncased')
	parser.add_argument("--coheren_model_name", default='bert-base-uncased')
	parser.add_argument("--max_len", type=int, default=512, help='Must match --max_len used at train time')

	args = parser.parse_args()
	if args.single_ckpt:
		assert args.ckpt, "--ckpt is required when --single_ckpt is set"
	args.dataset = DATASET[args.dataset]
	args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
	os.makedirs(f'{args.root}/metric/{args.model}', exist_ok=True)
	set_seed(3407)
	if args.single_ckpt:
		model_path = args.ckpt
		infer(args, model_path)
	else:
		# skip metrics.json / loss.json that train.py writes next to the checkpoints
		ckpts = sorted(f for f in os.listdir(args.root+'/model/'+args.model) if f.endswith('.pt'))[args.ckpt_start:args.ckpt_end]
		save_prefix = args.save_name
		for ckpt in tqdm(ckpts):
			MODEL_PATH = args.root+'/model/'+args.model +'/'+ckpt
			args.save_name = f'{save_prefix}_{os.path.splitext(ckpt)[0]}'
			infer(args, MODEL_PATH)
			

	