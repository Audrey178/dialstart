import os
import torch
import segeval
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from keras.preprocessing.sequence import pad_sequences
from model import encode_coherence_pair


def depth_score_cal(scores):
    output_scores = []
    for i in range(len(scores)):
        lflag, rflag = scores[i], scores[i]
        if i == 0:
            for r in range(i+1, len(scores)):
                if rflag <= scores[r]:
                    rflag = scores[r]
                else:
                    break
        elif i == len(scores)-1:
            for l in range(i-1, -1, -1):
                if lflag <= scores[l]:
                    lflag = scores[l]
                else:
                    break
        else:
            for r in range(i+1, len(scores)):
                if rflag <= scores[r]:
                    rflag = scores[r]
                else:
                    break
            for l in range(i-1, -1, -1):
                if lflag <= scores[l]:
                    lflag = scores[l]
                else:
                    break
        depth_score = 0.5*(lflag+rflag-2*scores[i])
        output_scores.append(depth_score)
    return output_scores


@torch.no_grad()
def evaluate_dataset(model, tokenizer, topic_tokenizer, path_input_docs, device, max_len,
                      window_size=2, oracle_boundary_count=True, pick_num=4, max_docs=None,
                      infer_batch_size=64):
    """Score every document under path_input_docs with `model` and return (pk, wd).

    Shared by test.py (standalone eval of a saved checkpoint) and train.py
    (in-loop validation after each epoch), so both always score boundaries
    the same way. `tokenizer` feeds the coherence encoder and `topic_tokenizer`
    (a model.TopicTokenizer) the topic encoder. max_docs scores only the first N files (sorted), e.g. to
    track train-set pk cheaply. infer_batch_size caps how many utterance pairs
    go through the model per forward (long meetings would not fit at once).
    """
    was_training = model.training
    model.eval()

    input_files = sorted(f for f in os.listdir(path_input_docs) if os.path.isfile(os.path.join(path_input_docs, f)))
    if max_docs is not None:
        input_files = input_files[:max_docs]

    c = score_wd = score_pk = 0
    for file in input_files:
        if file in ['.DS_Store']:
            continue

        text, id_inputs, coheren_att_masks, type_ids = [], [], [], []
        topic_input, topic_att_mask, topic_num = [[], []], [[], []], [[], []]
        seg_r_labels, seg_r = [], []
        tmp = 0
        for line in open(os.path.join(path_input_docs, file)):
            if '================' not in line.strip():
                text.append(line.strip())
                seg_r_labels.append(0)
                tmp += 1
            else:
                seg_r_labels[-1] = 1
                seg_r.append(tmp)
                tmp = 0
        seg_r.append(tmp)

        for i in range(len(text)-1):
            context, cur = [], []
            l, r = i, i+1
            # Utterances go in whole: both tokenizers truncate by tokens below,
            # the same way training does (no character cut, which dropped half
            # of the text of long meeting turns).
            for win in range(window_size):
                if l > -1:
                    context.append(text[l])
                    l -= 1
                if r < len(text):
                    cur.append(text[r])
                    r += 1
            context.reverse()

            topic_con = topic_tokenizer(context, max_length=256, padding=True, return_tensors='pt')
            topic_cur = topic_tokenizer(cur, max_length=256, padding=True, return_tensors='pt')

            topic_input[0].extend(topic_con['input_ids'])
            topic_input[1].extend(topic_cur['input_ids'])
            topic_att_mask[0].extend(topic_con['attention_mask'])
            topic_att_mask[1].extend(topic_cur['attention_mask'])
            topic_num[0].append(len(context))
            topic_num[1].append(len(cur))

            encoded_pair, type_id = encode_coherence_pair(tokenizer, context, text[i+1], max_len)
            type_ids.append(torch.Tensor(type_id))
            id_inputs.append(torch.Tensor(encoded_pair))

        id_inputs = pad_sequences(id_inputs, maxlen=max_len, dtype="long", value=0, truncating="post", padding="post")
        type_ids = pad_sequences(type_ids, maxlen=max_len, dtype="long", value=1, truncating="post", padding="post")
        for sent in id_inputs:
            att_mask = [int(token_id > 0) for token_id in sent]
            coheren_att_masks.append(att_mask)

        if not topic_num[0]:
            continue

        # Pairs are scored independently, so scoring them in chunks gives the
        # same result as one forward over the whole document while keeping
        # memory bounded for long meetings.
        id_inputs, coheren_att_masks, type_ids = torch.tensor(id_inputs), torch.tensor(coheren_att_masks), torch.tensor(type_ids)
        utt_offsets = [np.cumsum([0] + nums) for nums in topic_num]
        scores = []
        for start in range(0, len(topic_num[0]), infer_batch_size):
            end = min(start + infer_batch_size, len(topic_num[0]))
            topic_input_t, topic_mask_t = [], []
            for k in range(2):
                lo, hi = utt_offsets[k][start], utt_offsets[k][end]
                topic_input_t.append(pad_sequence(topic_input[k][lo:hi], batch_first=True,
                                                  padding_value=topic_tokenizer.pad_token_id).to(device))
                topic_mask_t.append(pad_sequence(topic_att_mask[k][lo:hi], batch_first=True).to(device))
            chunk_num = [topic_num[0][start:end], topic_num[1][start:end]]

            # Drop columns that are padding for every pair (masked anyway).
            coheren_len = max(1, int(coheren_att_masks[start:end].sum(dim=1).max()))
            scores += model.infer(id_inputs[start:end, :coheren_len].to(device),
                                  coheren_att_masks[start:end, :coheren_len].to(device),
                                  type_ids[start:end, :coheren_len].to(device),
                                  topic_input_t, topic_mask_t, chunk_num)

        depth_scores = depth_score_cal(scores)
        pick_n = (len(seg_r) - 1) if oracle_boundary_count else pick_num
        boundary_indice = np.argsort(np.array(depth_scores))[-pick_n:] if pick_n > 0 else np.array([], dtype=int)
        seg_p_labels = [0]*(len(depth_scores)+1)
        for i in boundary_indice:
            seg_p_labels[i] = 1

        tmp = 0
        seg_p = []
        for fake in seg_p_labels:
            if fake == 1:
                tmp += 1
                seg_p.append(tmp)
                tmp = 0
            else:
                tmp += 1
        seg_p.append(tmp)

        score_wd += segeval.window_diff(seg_p, seg_r)
        score_pk += segeval.pk(seg_p, seg_r)
        c += 1

    if was_training:
        model.train()

    if c == 0:
        return None, None
    return float(score_pk/c), float(score_wd/c)
