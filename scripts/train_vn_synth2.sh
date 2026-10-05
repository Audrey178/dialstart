# P = data prefix: vn3 (default, shortcut-free assembly) or vn2 (first build).
P=${P:-vn3}
# TOPIC / TAG must match the values used for scripts/preprocess_vn_synth2.sh.
TOPIC=${TOPIC:-VoVanPhuc/sup-SimCSE-VietNamese-phobert-base}
TAG=${TAG-_phobert}
# Needs data/${P}_train${TAG}.pkl: run scripts/preprocess_vn_synth2.sh first.
# Early stopping / best.pt use ${P}_val_in; out-of-domain splits are scored by test_vn_synth2.sh.
set -e
if [ ! -f ./data/${P}_train${TAG}.pkl ]; then
    echo "data/${P}_train${TAG}.pkl not found, run: bash scripts/preprocess_vn_synth2.sh" >&2
    exit 1
fi

python ./train.py --dataset ${P}_train --data_name "$TAG" \
    --save_model_name model_${P}${TAG} \
    --topic_model_name $TOPIC \
    --coheren_model_name NlpHUST/vibert4news-base-cased \
    --batch_size 32 --accum 1 --epoch 5 --patience 2 \
    --val_dataset ${P}_val_in --eval_max_len 160 --eval_oracle_boundary_count \
    --train_eval_dataset ${P}_train --train_eval_docs 100
