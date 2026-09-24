# Scores best.pt (picked on vn2_val_in) on every vn2 eval split.
# Each split writes metric/model_vn2/<split>.json.
CKPT=./model/model_vn2/best.pt
if [ ! -f "$CKPT" ]; then
    echo "$CKPT not found, run: bash scripts/train_vn2.sh" >&2
    exit 1
fi

for SPLIT in vn2_val_in vn2_test_in vn2_val_ood vn2_test_ood; do
    python test.py --model model_vn2 --dataset $SPLIT --save_name $SPLIT \
        --single_ckpt --ckpt $CKPT \
        --topic_model_name NlpHUST/vibert4news-base-cased \
        --coheren_model_name NlpHUST/vibert4news-base-cased \
        --max_len 160 --oracle_boundary_count
done
