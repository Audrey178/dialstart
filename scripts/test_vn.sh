# Scores the checkpoint train.py kept (best val pk), not every epoch.
# Each split writes its own metric/model_vn/<split>.json.
CKPT=./model/model_vn/best.pt
if [ ! -f "$CKPT" ]; then
    echo "$CKPT not found, run: bash scripts/train_vn.sh" >&2
    exit 1
fi

for SPLIT in vn_val vn_test; do
    python test.py --model model_vn --dataset $SPLIT --save_name $SPLIT \
        --single_ckpt --ckpt $CKPT \
        --topic_model_name NlpHUST/vibert4news-base-cased \
        --coheren_model_name NlpHUST/vibert4news-base-cased \
        --max_len 160 --oracle_boundary_count
done
