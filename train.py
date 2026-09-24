import re
import os
import math
import json
import torch
import random
import pickle
import argparse
import torch.nn as nn
from tqdm import tqdm
from model import SegModel
from eval_utils import evaluate_dataset
from torch.cuda import amp
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, Dataset
from transformers import BertForNextSentencePrediction, BertConfig, BertTokenizer, get_linear_schedule_with_warmup, set_seed, AutoModel
from torch.optim import AdamW

DATASET = {'doc':'doc2dial', '711':'dialseg711', 'vn':'vn_synth_train', 'vn2':'vn2_train'}
def get_mask(tensor):
    attention_masks = []
    for sent in tensor:
        att_mask = [int(token_id > 0) for token_id in sent]
        attention_masks.append(att_mask)
    return torch.tensor(attention_masks)


class ourdataset(Dataset):
    def __init__(self, loaded_data):
        self.loaded_data = loaded_data
        
    def __getitem__(self, idx):
        return [i[idx] for i in self.loaded_data]
   
    def __len__(self):
        return len(self.loaded_data[0])

    def collect_fn(self, examples):
        batch_size, topic_train, topic_train_mask, topic_num = len(examples), torch.tensor(0), torch.tensor(0), torch.tensor(0)
        coheren_inputs = pad_sequence([ex[0] for ex in examples], batch_first=True)
        coheren_mask = pad_sequence([ex[1] for ex in examples], batch_first=True)
        coheren_type = pad_sequence([ex[2] for ex in examples], batch_first=True)
        # Pairs are stored post-padded to --max_len; cut the columns that are
        # padding for every pair in the batch (masked anyway, so same output).
        coheren_len = int(coheren_mask.sum(-1).max())
        coheren_inputs, coheren_mask, coheren_type = coheren_inputs[..., :coheren_len], \
            coheren_mask[..., :coheren_len], coheren_type[..., :coheren_len]

        topic_context = pad_sequence([torch.tensor(j) for ex in examples for j in ex[3][0][0]['input_ids']], batch_first=True)
        topic_pos = pad_sequence([torch.tensor(j) for ex in examples for j in ex[3][0][1]['input_ids']], batch_first=True)
        topic_neg = pad_sequence([torch.tensor(j) for ex in examples for j in ex[3][1][1]['input_ids']], batch_first=True) #TODO

        topic_context_num = [ex[3][0][2] for ex in examples]
        topic_pos_num = [ex[3][0][3] for ex in examples]
        topic_neg_num = [ex[3][1][3] for ex in examples]

        topic_context_mask, topic_pos_mask, topic_neg_mask = get_mask(topic_context), get_mask(topic_pos), get_mask(topic_neg)

        topic_train = pad_sequence([j for ex in examples for j in ex[4]], batch_first=True)
        topic_train_mask = pad_sequence([j for ex in examples for j in ex[5]], batch_first=True)
        topic_num = [ex[6] for ex in examples]

            
        return coheren_inputs, coheren_mask, coheren_type, topic_context, topic_pos, topic_neg, \
            topic_context_mask, topic_pos_mask, topic_neg_mask, \
            topic_context_num, topic_pos_num, topic_neg_num, \
            topic_train, topic_train_mask, topic_num
        


def main(args):
    print(f"Loading data from {args.root}data/{DATASET[args.dataset]}{args.data_name}.pkl")
    loaded_data = pickle.load(open(f'{args.root}/data/{DATASET[args.dataset]}{args.data_name}.pkl', 'rb'))
    
    epochs = args.epoch
    global_step = continue_from_global_step = 0
    train_data = ourdataset(loaded_data)

    out_path = f'{args.root}/model/{args.save_model_name}'
    val_tokenizer, val_path, train_eval_path = None, None, None
    if args.val_dataset or args.train_eval_dataset:
        val_tokenizer = BertTokenizer.from_pretrained(args.coheren_model_name)
    if args.val_dataset:
        val_path = f'{args.root}/data/{args.val_dataset}'
    if args.train_eval_dataset:
        train_eval_path = f'{args.root}/data/{args.train_eval_dataset}'
     
    train_sampler = RandomSampler(train_data) if args.local_rank == -1 else DistributedSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.batch_size, collate_fn=train_data.collect_fn)

    scaler = amp.GradScaler(enabled=(not args.no_amp))
    model = SegModel(margin=args.margin, train_split=args.train_split, window_size=args.window_size,
                      topic_model_name=args.topic_model_name, coheren_model_name=args.coheren_model_name,
                      gradient_checkpointing=args.grad_checkpoint).to(args.device)
    if args.resume:
        # continue_from_global_step = len(train_dataloader) * (int(args.ckpt.split('/')[-1]) + 1)
        # continue_from_global_step = int(args.ckpt.split('-')[-1])
        model.load_state_dict(torch.load(f'{args.root}/model/{args.ckpt}'), False)
    if args.local_rank != -1:
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

    epoch_loss = {}
    epoch_metrics = {}
    if args.optim == 'adamw8bit':
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=args.lr, eps=1e-8)
    else:
        optimizer = AdamW(model.parameters(), lr=args.lr, eps = 1e-8)
    # The scheduler steps once per optimizer update, i.e. every `accum` batches.
    total_steps = math.ceil(len(train_dataloader) / args.accum) * epochs
    num_warmup_steps = int(total_steps * args.warmup_proportion)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps = total_steps)

    best_pk, best_epoch, bad_epochs = None, None, 0
    for epoch_i in tqdm(range(epochs)):
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        total_loss = total_margin_loss = total_topic_loss = 0
        model.train()
        epoch_iterator = tqdm(train_dataloader, disable=args.local_rank!=-1)
        window_size = args.window_size

        for step, batch in enumerate(epoch_iterator):
            if global_step < continue_from_global_step:
                if (step + 1) % args.accum == 0:
                    scheduler.step()
                    global_step += 1
                continue

            input_data = {'coheren_inputs' : batch[0].to(args.device),
                            'coheren_mask' : batch[1].to(args.device),
                            'coheren_type' : batch[2].to(args.device),
                            'topic_context' : batch[3].to(args.device),
                            'topic_pos' : batch[4].to(args.device),
                            'topic_neg' : batch[5].to(args.device),
                            'topic_context_mask' : batch[6].to(args.device),
                            'topic_pos_mask' : batch[7].to(args.device),
                            'topic_neg_mask' : batch[8].to(args.device),
                            'topic_context_num' : batch[9],
                            'topic_pos_num' : batch[10],
                            'topic_neg_num' : batch[11],
                            'topic_train' : batch[12].to(args.device),
                            'topic_train_mask' : batch[13].to(args.device),
                            'topic_num' : batch[14]
                            }

            with amp.autocast(enabled=(not args.no_amp)):
                loss, margin_loss, topic_loss = model(input_data, window_size)

            if args.n_gpu > 1:
                loss = loss.mean()

            total_loss += loss.item()
            total_margin_loss += margin_loss.mean().item()
            total_topic_loss += topic_loss.mean().item()
            if (not args.no_amp):
                scaler.scale(loss).backward()
                if (step + 1) % args.accum == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    model.zero_grad()
                    scheduler.step()
                    global_step += 1
            else:
                loss.backward()
                if (step + 1) % args.accum == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    model.zero_grad()
                    scheduler.step()
                    global_step += 1
            
        num_batches = len(train_dataloader)
        avg_train_loss = total_loss / num_batches
        epoch_loss[epoch_i] = avg_train_loss
        epoch_metrics[epoch_i] = {'loss': avg_train_loss,
                                  'margin_loss': total_margin_loss / num_batches,
                                  'topic_loss': total_topic_loss / num_batches}
        stop_training = False
        if args.local_rank in [-1, 0]:
            print('=========== the loss for epoch '+str(epoch_i)+' is: '+str(avg_train_loss)
                  +' (margin '+str(epoch_metrics[epoch_i]['margin_loss'])
                  +', topic '+str(epoch_metrics[epoch_i]['topic_loss'])+')')
            model_to_save = model.module if hasattr(model, 'module') else model

            eval_kwargs = dict(window_size=args.eval_window_size,
                               oracle_boundary_count=args.eval_oracle_boundary_count,
                               pick_num=args.eval_pick_num)
            run_eval = (epoch_i + 1) % args.eval_every == 0

            # pk on a slice of the training sessions: a train pk that keeps
            # improving while val pk stalls means the model is memorising.
            if train_eval_path and run_eval:
                train_pk, train_wd = evaluate_dataset(model_to_save, val_tokenizer, train_eval_path, args.device,
                                                       args.eval_max_len, max_docs=args.train_eval_docs, **eval_kwargs)
                print('=========== train pk/wd for epoch '+str(epoch_i)+' is: '+str(train_pk)+' / '+str(train_wd))
                epoch_metrics[epoch_i]['train_pk'] = train_pk
                epoch_metrics[epoch_i]['train_wd'] = train_wd

            if val_path and run_eval:
                pk, wd = evaluate_dataset(model_to_save, val_tokenizer, val_path, args.device, args.eval_max_len,
                                           **eval_kwargs)
                print('=========== val pk/wd for epoch '+str(epoch_i)+' is: '+str(pk)+' / '+str(wd))
                epoch_metrics[epoch_i]['val_pk'] = pk
                epoch_metrics[epoch_i]['val_wd'] = wd

                # Keep only the checkpoint with the best val pk, and stop once
                # it has not improved for `patience` evaluations.
                if pk is not None and (best_pk is None or pk < best_pk):
                    best_pk, best_epoch, bad_epochs = pk, epoch_i, 0
                    PATH = f'{out_path}/best.pt'
                    print('Saving best model (val pk '+str(pk)+') to '+PATH)
                    torch.save(model_to_save.state_dict(), PATH)
                else:
                    bad_epochs += 1
                    if args.patience > 0 and bad_epochs >= args.patience:
                        print('=========== val pk has not improved for '+str(bad_epochs)+' evals, stopping early')
                        stop_training = True
            elif not val_path:
                PATH = f'{out_path}/{str(epoch_i)}-{str(global_step)}'
                print('Saving model to '+ PATH)
                torch.save(model_to_save.state_dict(), PATH)

            if args.save_every_epoch and val_path:
                torch.save(model_to_save.state_dict(), f'{out_path}/{str(epoch_i)}-{str(global_step)}')

            epoch_metrics['best'] = {'epoch': best_epoch, 'val_pk': best_pk}
            json.dump(epoch_metrics, open(f'{out_path}/metrics.json', 'w'), indent=2)
            model.train()

        if stop_training:
            break
    return epoch_loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--save_model_name", required=True)
    # model parameters
    parser.add_argument("--margin", type=int, default=1)
    parser.add_argument("--train_split", type=int, default=5)
    parser.add_argument("--window_size", type=int, default=5)
    parser.add_argument("--topic_model_name", default='princeton-nlp/sup-simcse-bert-base-uncased')
    parser.add_argument("--coheren_model_name", default='bert-base-uncased')
    parser.add_argument("--grad_checkpoint", action='store_true', help='Trade compute for activation memory, for low-VRAM GPUs')

    # path parameters
    parser.add_argument("--ckpt")
    parser.add_argument("--data_name", default='')
    parser.add_argument("--root", default='.')
    parser.add_argument("--epoch", type=int, default=10, help='Number of training epochs')
    parser.add_argument("--seed", type=int, default=3407)

    # in-training validation parameters
    parser.add_argument("--val_dataset", default=None,
                         help='Folder name under {root}/data to score pk/wd on after each epoch (e.g. vn_synth_val). Omit to skip in-training validation.')
    parser.add_argument("--eval_every", type=int, default=1, help='Run validation every N epochs')
    parser.add_argument("--eval_max_len", type=int, default=512,
                         help='Padding length for validation coherence pairs; should match --max_len used for preprocessing/testing')
    parser.add_argument("--eval_window_size", type=int, default=2,
                         help='Context window (utterances each side) used when scoring validation boundaries; matches test.py --window_size')
    parser.add_argument("--eval_pick_num", type=int, default=4)
    parser.add_argument("--eval_oracle_boundary_count", action='store_true')
    parser.add_argument("--train_eval_dataset", default=None,
                         help='Folder name under {root}/data of raw training .txt sessions (e.g. vn_synth_train); pk is scored on the first --train_eval_docs of them each epoch to expose memorisation')
    parser.add_argument("--train_eval_docs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=2,
                         help='Stop after this many evaluations without val pk improvement (0 disables early stopping)')
    parser.add_argument("--save_every_epoch", action='store_true',
                         help='With --val_dataset, also keep a checkpoint per epoch besides best.pt')

    # train parameters
    parser.add_argument('--accum', type=int, default=1)
    parser.add_argument("--resume", action='store_true')
    parser.add_argument("--lr", default=3e-5, type=float)
    parser.add_argument("--optim", default='adamw', choices=['adamw', 'adamw8bit'],
                         help='adamw8bit uses bitsandbytes to cut optimizer state memory ~4x, for low-VRAM GPUs')
    parser.add_argument("--batch_size", default=12, type=int)
    parser.add_argument("--warmup_proportion", default=0.1, type=float)
    
    #device parameters
    parser.add_argument("--no_amp", action='store_true')
    parser.add_argument("--no_cuda", action='store_true')
    parser.add_argument("--local_rank", type=int, default=-1)
    
    args = parser.parse_args()
    set_seed(args.seed)
    
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = torch.cuda.device_count()
    
    args.device = device
    out_path = f'{args.root}/model/{args.save_model_name}'
    os.makedirs(out_path, exist_ok=True) 
    epoch_loss = main(args)
    json.dump(epoch_loss, open(f'{out_path}/loss.json', 'w'))














