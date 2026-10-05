# TOPIC / TAG / MAX_LEN must match the values used for scripts/train_dialstart_vi.sh.
TOPIC=${TOPIC:-VoVanPhuc/sup-SimCSE-VietNamese-phobert-base}
TAG=${TAG-_phobert}
MAX_LEN=${MAX_LEN:-192}
# MODEL = model dir to score; fine-tuned runs are model_dialstart_vi${TAG}_from_<init>.
MODEL=${MODEL:-model_dialstart_vi${TAG}}
# Scores best.pt (picked on dev) on dev and test.
# Each split writes metric/${MODEL}/<split>.json.
CKPT=./model/${MODEL}/best.pt
if [ ! -f "$CKPT" ]; then
    echo "$CKPT not found, run: bash scripts/train_dialstart_vi.sh" >&2
    exit 1
fi

for SPLIT in dev test; do
    python test.py --model ${MODEL} --dataset dialstart_vi/$SPLIT --save_name $SPLIT \
        --single_ckpt --ckpt $CKPT \
        --topic_model_name $TOPIC \
        --coheren_model_name NlpHUST/vibert4news-base-cased \
        --max_len $MAX_LEN --oracle_boundary_count --infer_batch_size 32
done
