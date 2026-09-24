# vn_synth v2 (scripts/synth pipeline). Run once per data version, after
# scripts/check_data_stats.py prints 0 FAIL on the vn2_* folders.
python ./data_preprocess.py \
    --datasets vn2_train \
    --encoder_name NlpHUST/vibert4news-base-cased \
    --max_len 160
