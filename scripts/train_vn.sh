# Needs data/vn_synth_train.pkl: run scripts/preprocess_vn.sh first (once per data version).
# H200 settings: no --grad_checkpoint / --optim adamw8bit, those only save
# memory on small GPUs and cost speed.
set -e
if [ ! -f ./data/vn_synth_train.pkl ]; then
    echo "data/vn_synth_train.pkl not found, run: bash scripts/preprocess_vn.sh" >&2
    exit 1
fi

python ./train.py --dataset vn \
    --save_model_name model_vn \
    --topic_model_name NlpHUST/vibert4news-base-cased \
    --coheren_model_name NlpHUST/vibert4news-base-cased \
    --batch_size 32 --accum 1 --epoch 5 --patience 2 \
    --val_dataset vn_synth_val --eval_max_len 160 --eval_oracle_boundary_count \
    --train_eval_dataset vn_synth_train --train_eval_docs 100
