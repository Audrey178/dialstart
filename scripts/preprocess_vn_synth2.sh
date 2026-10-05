# P = data prefix: vn3 (default, shortcut-free assembly) or vn2 (first build).
P=${P:-vn3}
# TOPIC = topic encoder, TAG = suffix of the pkl / model dir. Baseline (vibert4news for both):
#   TOPIC=NlpHUST/vibert4news-base-cased TAG= bash scripts/preprocess_vn_synth2.sh
TOPIC=${TOPIC:-VoVanPhuc/sup-SimCSE-VietNamese-phobert-base}
TAG=${TAG-_phobert}
# vn_synth v2 (scripts/synth pipeline). Run once per data version, after
# scripts/check_data_stats.py prints 0 FAIL on the ${P}_* folders.
python ./data_preprocess.py \
    --datasets ${P}_train \
    --encoder_name NlpHUST/vibert4news-base-cased \
    --topic_encoder_name $TOPIC \
    --save_name "$TAG" \
    --max_len 160
