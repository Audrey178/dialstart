import os
import re
import json
import torch
import random
import pickle
import IPython
import argparse
import subprocess
import numpy as np
from tqdm import tqdm
from torch.nn import CrossEntropyLoss
from collections import defaultdict, Counter
from torch.nn.utils.rnn import pad_sequence
from keras.preprocessing.sequence import pad_sequences
from transformers import BertTokenizer, AutoModel, set_seed, BertForNextSentencePrediction
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, Dataset

set_seed(3407)
DATASET = {'doc':'doc2dial', '711':'dialseg711'}


def gen_text(args):
    data, topic_data = [], []
    w = 2
    for dataset in args.datasets:
        todolist = [f'{args.dataroot}/{dataset}/'+i for i in os.listdir(f'{args.dataroot}/{dataset}') if not i.startswith('.')]

        for i in tqdm(todolist):
            dial_name = i.split('/')[-1][:-4]
            cur_dials = open(i).read().split('\n')[:-1]

            # Recover per-utterance segment id from the "====" boundary markers
            # instead of discarding them, so negatives can be drawn from a
            # genuinely different topic segment rather than a fixed distance
            # (which, for short segments, often lands back inside the same
            # segment as the anchor and injects false negatives).
            dials, seg_id = [], []
            cur_seg = 0
            for utt in cur_dials:
                if '=======' in utt:
                    cur_seg += 1
                else:
                    dials.append(utt)
                    seg_id.append(cur_seg)
            dial_len = len(dials)

            seg_to_indices = defaultdict(list)
            for idx, s in enumerate(seg_id):
                seg_to_indices[s].append(idx)

            for utt_idx in range(dial_len-1):
                context, cur, neg, hard_neg = [], [], [], []
                anchor_seg = seg_id[utt_idx]
                cross_seg_pool = [j for s, idxs in seg_to_indices.items() if s != anchor_seg for j in idxs]

                if cross_seg_pool:
                    # Genuine negative: an utterance from a different topic
                    # segment of the SAME dialogue. Two independent draws
                    # replace the old "same-dialogue distance-w" negative and
                    # the old "random other dialogue" hard negative, both of
                    # which let the model shortcut on domain/vocabulary cues
                    # instead of learning within-document topic boundaries.
                    neg_index = random.choice(cross_seg_pool)
                    neg_hard_index = random.choice(cross_seg_pool)
                else:
                    # Single-segment dialogue: no cross-boundary utterance
                    # exists, fall back to the old distance-based sampling.
                    fallback_pool = list(range(utt_idx-w+1)) + list(range(utt_idx+w+1, dial_len))
                    neg_index = random.choice(fallback_pool) if fallback_pool else utt_idx
                    neg_hard_index = neg_index

                mid = utt_idx+1
                l, r = utt_idx, utt_idx+1
                for hi in range(args.history):
                    if l > -1:
                        context.append(re.sub(r'\s([,?.!"](?:\s|$))', r'\1', dials[l]))
                        l -= 1
                    if r < dial_len:
                        cur.append(re.sub(r'\s([,?.!"](?:\s|$))', r'\1', dials[r]))
                        r += 1
                    if neg_index < dial_len:
                        neg.append(re.sub(r'\s([,?.!"](?:\s|$))', r'\1', dials[neg_index]))
                        neg_index += 1
                    if neg_hard_index < dial_len:
                        hard_neg.append(re.sub(r'\s([,?.!"](?:\s|$))', r'\1', dials[neg_hard_index]))
                        neg_hard_index += 1

                context.reverse()
                data.append([(context, cur), (context, neg), (context, hard_neg)])
                topic_data.append((dials, mid))

                assert len(dials) > mid

        json.dump(data, open(f'{args.dataroot}/{dataset}_{args.version}.json', 'w'))
        json.dump(topic_data, open(f'{args.dataroot}/{dataset}_topic_data.json', 'w'))


def main(args):
    MAX_LEN = args.max_len
    tokenizer = BertTokenizer.from_pretrained(args.encoder_name)
    for dataset in tqdm(args.datasets):
        data = json.load(open(f'{args.dataroot}/{dataset}_{args.version}.json'))
        topic_data = json.load(open(f'{args.dataroot}/{dataset}_topic_data.json'))
        if args.limit is not None:
            data = data[:args.limit]
            topic_data = topic_data[:args.limit]

        turn_ids, id_inputs, topic_inputs, sample_num_memory, topic_train, topic_num = [], [], [], [len(i) for i in data], [], []
        for i in tqdm(range(len(data))):
            for sample in data[i]:
                context, cur = sample
                if args.history == 2:
                    sent1 = sent2 = ''
                    for sen in context:
                        sent1 += sen + '[SEP]'
                else:
                    sent1 = context[-1]

                sent2 = cur[0]

                encoded_sent1 = tokenizer.encode(sent1, add_special_tokens = True, return_tensors = 'pt')
                encoded_sent2 = tokenizer.encode(sent2, truncation=True, max_length = 256, add_special_tokens = True, return_tensors = 'pt')

                topic_con = tokenizer(context, truncation=True, max_length = 256)
                topic_cur = tokenizer(cur, truncation=True, max_length = 256)
                if args.history == 2:
                    id_input = encoded_sent1[0].tolist()[-257:-1] + encoded_sent2[0].tolist()[1:]  
                    turn_id = [0] * len(encoded_sent1[0].tolist()[-257:-1]) + [1] * len(encoded_sent2[0].tolist()[1:])
                else:
                    id_input = encoded_sent1[0].tolist()[-257:] + encoded_sent2[0].tolist()[1:]
                    turn_id = [0] * len(encoded_sent1[0].tolist()[-257:]) + [1] * len(encoded_sent2[0].tolist()[1:])

                id_inputs.append(torch.Tensor(id_input))
                topic_inputs.append((topic_con, topic_cur, len(context), len(cur)))
                turn_ids.append(torch.tensor(turn_id))

            topic_train.append(tokenizer(topic_data[i][0], truncation=True, max_length = 512, padding=True, return_tensors='pt'))
            topic_num.append((len(topic_data[i][0]), topic_data[i][1])) 

        id_inputs = pad_sequences(id_inputs, maxlen=MAX_LEN, dtype="long", value=0, truncating="post", padding="post")
        turn_ids = pad_sequences(turn_ids, maxlen=MAX_LEN, dtype="long", value=1, truncating="post", padding="post")

        topic_train_input = [i['input_ids'] for i in topic_train]
        topic_train_mask = [i['attention_mask'] for i in topic_train]
        attention_masks = []
        for sent in tqdm(id_inputs):
            att_mask = [int(token_id > 0) for token_id in sent]
            attention_masks.append(att_mask)

        grouped_inputs, grouped_masks, grouped_topic, grouped_token_type_id = [], [], [], []
        token_type_id = []
        count = 0
        for i in tqdm(sample_num_memory):
            grouped_inputs.append(id_inputs[count: count+i])
            grouped_masks.append(attention_masks[count: count+i])
            grouped_topic.append(topic_inputs[count: count+i])
            grouped_token_type_id.append(turn_ids[count:count+i])
            count += i

        pos_neg_pairs, pos_neg_masks, pos_neg_token_types, topic_pairs = [], [], [], []
        topic_trains, topic_trains_mask, topic_nums = [], [], []
        for i in tqdm(range(len(grouped_inputs))):
            if len(grouped_inputs[i]) == 2:
                pos_neg_pairs.append(grouped_inputs[i])
                pos_neg_token_types.append(grouped_token_type_id[i])
                pos_neg_masks.append(grouped_masks[i])
            else:
                topic_pairs.append([grouped_topic[i][0], grouped_topic[i][1]])
                topic_pairs.append([grouped_topic[i][0], grouped_topic[i][2]])
                
                topic_trains.append(topic_train_input[i])
                topic_trains.append(topic_train_input[i])
                topic_trains_mask.append(topic_train_mask[i])
                topic_trains_mask.append(topic_train_mask[i])
                topic_nums.append(topic_num[i])
                topic_nums.append(topic_num[i])

                pos_neg_pairs.append([grouped_inputs[i][0], grouped_inputs[i][1]])
                pos_neg_pairs.append([grouped_inputs[i][0], grouped_inputs[i][2]])

                pos_neg_token_types.append([grouped_token_type_id[i][0], grouped_token_type_id[i][1]])
                pos_neg_token_types.append([grouped_token_type_id[i][0], grouped_token_type_id[i][2]])

                pos_neg_masks.append([grouped_masks[i][0], grouped_masks[i][1]])
                pos_neg_masks.append([grouped_masks[i][0], grouped_masks[i][2]])

        train_inputs, train_masks, train_types = torch.tensor(pos_neg_pairs), torch.tensor(pos_neg_masks), torch.tensor(pos_neg_token_types)
        pickle.dump((train_inputs, train_masks, train_types, topic_pairs, topic_trains, topic_trains_mask, topic_nums), 
                    open(f'{args.dataroot}/{dataset}{args.save_name}.pkl', 'wb'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_name", default='', help='The name of preprocessed data')
    parser.add_argument("--dataroot", default='./data')
    parser.add_argument("--history", type=int, default=2)
    parser.add_argument("--version", default='2h2cur')
    parser.add_argument("--datasets", nargs='+', default=['dialseg711', 'doc2dial'],
                         help='Folder name(s) under --dataroot to preprocess, e.g. vn_synth_train')
    parser.add_argument("--encoder_name", default='bert-base-uncased',
                         help='HF checkpoint used to tokenize; must match the encoder(s) SegModel is trained with')
    parser.add_argument("--limit", type=int, default=None,
                         help='Optional cap on number of dialogue-window samples per dataset (for smoke tests)')
    parser.add_argument("--max_len", type=int, default=512,
                         help='Fixed padding length for coherence input pairs; lower for corpora with short utterances to save memory/compute')

    args = parser.parse_args()

    gen_text(args)
    main(args)