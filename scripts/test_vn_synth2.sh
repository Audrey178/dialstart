# P = data prefix: vn3 (default, shortcut-free assembly) or vn2 (first build).
P=${P:-vn3}
# TOPIC / TAG must match the values used for scripts/train_vn_synth2.sh.
TOPIC=${TOPIC:-VoVanPhuc/sup-SimCSE-VietNamese-phobert-base}
TAG=${TAG-_phobert}
# Scores best.pt (picked on ${P}_val_in) on every ${P} eval split.
# Each split writes metric/model_${P}${TAG}/<split>.json.
CKPT=./model/model_${P}${TAG}/best.pt
if [ ! -f "$CKPT" ]; then
    echo "$CKPT not found, run: bash scripts/train_vn_synth2.sh" >&2
    exit 1
fi

for SPLIT in ${P}_val_in ${P}_test_in ${P}_val_ood ${P}_test_ood; do
    python test.py --model model_${P}${TAG} --dataset $SPLIT --save_name $SPLIT \
        --single_ckpt --ckpt $CKPT \
        --topic_model_name $TOPIC \
        --coheren_model_name NlpHUST/vibert4news-base-cased \
        --max_len 160 --oracle_boundary_count
done
