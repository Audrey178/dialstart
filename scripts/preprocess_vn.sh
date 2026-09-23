# Run once per data version (after regenerating data/vn_synth_train/), not
# before every training run: it is single-threaded and writes a multi-GB pkl.
python ./data_preprocess.py \
    --datasets vn_synth_train \
    --encoder_name NlpHUST/vibert4news-base-cased \
    --max_len 160
