# Needs data/vn2_train.pkl: run scripts/preprocess_vn2.sh first.
# Early stopping / best.pt use vn2_val_in; out-of-domain splits are scored by test_vn2.sh.
set -e
if [ ! -f ./data/vn2_train.pkl ]; then
    echo "data/vn2_train.pkl not found, run: bash scripts/preprocess_vn2.sh" >&2
    exit 1
fi

python ./train.py --dataset vn2 \
    --save_model_name model_vn2 \
    --topic_model_name NlpHUST/vibert4news-base-cased \
    --coheren_model_name NlpHUST/vibert4news-base-cased \
    --batch_size 32 --accum 1 --epoch 5 --patience 2 \
    --val_dataset vn2_val_in --eval_max_len 160 --eval_oracle_boundary_count \
    --train_eval_dataset vn2_train --train_eval_docs 100
