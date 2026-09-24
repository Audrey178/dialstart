# P = data prefix: vn3 (default, shortcut-free assembly) or vn2 (first build).
P=${P:-vn3}
# vn_synth v2 (scripts/synth pipeline). Run once per data version, after
# scripts/check_data_stats.py prints 0 FAIL on the ${P}_* folders.
python ./data_preprocess.py \
    --datasets ${P}_train \
    --encoder_name NlpHUST/vibert4news-base-cased \
    --max_len 160
