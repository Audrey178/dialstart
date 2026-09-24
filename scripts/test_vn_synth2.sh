# P = data prefix: vn3 (default, shortcut-free assembly) or vn2 (first build).
P=${P:-vn3}
# Scores best.pt (picked on ${P}_val_in) on every ${P} eval split.
# Each split writes metric/model_${P}/<split>.json.
CKPT=./model/model_${P}/best.pt
if [ ! -f "$CKPT" ]; then
    echo "$CKPT not found, run: bash scripts/train_vn_synth2.sh" >&2
    exit 1
fi

for SPLIT in ${P}_val_in ${P}_test_in ${P}_val_ood ${P}_test_ood; do
    python test.py --model model_${P} --dataset $SPLIT --save_name $SPLIT \
        --single_ckpt --ckpt $CKPT \
        --topic_model_name NlpHUST/vibert4news-base-cased \
        --coheren_model_name NlpHUST/vibert4news-base-cased \
        --max_len 160 --oracle_boundary_count
done
