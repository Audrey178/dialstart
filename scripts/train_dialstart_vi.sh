# TOPIC / TAG / MAX_LEN must match the values used for scripts/preprocess_dialstart_vi.sh.
TOPIC=${TOPIC:-VoVanPhuc/sup-SimCSE-VietNamese-phobert-base}
TAG=${TAG-_phobert}
MAX_LEN=${MAX_LEN:-192}
# Defaults target a 4GB GPU (effective batch 32 = BS * ACCUM); on a bigger GPU
# use e.g. BS=32 ACCUM=1 GRAD_CKPT=.
BS=${BS:-8}
ACCUM=${ACCUM:-4}
GRAD_CKPT=${GRAD_CKPT---grad_checkpoint}
# Pseudo-boundaries per topic window: AMI has ~1 topic shift per 30-40 utterances,
# so a 64-utterance window (TOPIC_WINDOW=32) holds ~2. Tune on dev.
TRAIN_SPLIT=${TRAIN_SPLIT:-2}
# Topic loss: pseudo (DialSTART, unsupervised), gold_margin or gold_supcon (use the
# labelled "====" boundaries). TOPIC_WEIGHT scales the topic cosine next to the NSP
# logit. Non-default values are appended to the model dir name.
TOPIC_LOSS=${TOPIC_LOSS:-pseudo}
TOPIC_WEIGHT=${TOPIC_WEIGHT:-1}
RUN=$([ "$TOPIC_LOSS" != pseudo ] && echo "_$TOPIC_LOSS")$([ "$TOPIC_WEIGHT" != 1 ] && echo "_tw$TOPIC_WEIGHT")
# INIT = checkpoint under model/ to fine-tune from, e.g. INIT=model_vn3_phobert/best.pt
# (must use the same TOPIC encoder). Weights only: optimizer/LR schedule start fresh.
# Empty = start from the pretrained encoders.
INIT=${INIT:-}
INIT_ARGS=${INIT:+--resume --ckpt $INIT}
# Early stopping / best.pt use dialstart_vi/dev; test is scored by test_dialstart_vi.sh.
set -e
if [ ! -f ./data/dialstart_vi/train${TAG}.pkl ]; then
    echo "data/dialstart_vi/train${TAG}.pkl not found, run: bash scripts/preprocess_dialstart_vi.sh" >&2
    exit 1
fi

python ./train.py --dataset dialstart_vi/train --data_name "$TAG" \
    --save_model_name model_dialstart_vi${TAG}${RUN}${INIT:+_from_$(dirname $INIT | sed 's/^model_//')} $INIT_ARGS \
    --topic_model_name $TOPIC --topic_loss $TOPIC_LOSS --topic_weight $TOPIC_WEIGHT \
    --coheren_model_name NlpHUST/vibert4news-base-cased \
    --batch_size $BS --accum $ACCUM $GRAD_CKPT --train_split $TRAIN_SPLIT \
    --epoch 5 --patience 2 \
    --val_dataset dialstart_vi/dev --eval_max_len $MAX_LEN --eval_oracle_boundary_count --eval_batch_size 32 \
    --train_eval_dataset dialstart_vi/train --train_eval_docs 10 "$@"
