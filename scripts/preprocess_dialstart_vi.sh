# DialSTART-vi: AMI meetings translated to Vietnamese, data/dialstart_vi/{train,dev,test}.
# TOPIC = topic encoder, TAG = suffix of the pkl / model dir (same convention as preprocess_vn_synth2.sh).
TOPIC=${TOPIC:-VoVanPhuc/sup-SimCSE-VietNamese-phobert-base}
TAG=${TAG-_phobert}
# Meetings run up to ~1000 utterances, so the topic loss only sees +/- TOPIC_WINDOW
# utterances around each anchor; without it the pkl does not fit in RAM.
TOPIC_WINDOW=${TOPIC_WINDOW:-32}
# Token cap per utterance for that topic window (PhoBERT p99 ~132 on train).
TOPIC_MAX_LEN=${TOPIC_MAX_LEN:-128}
# Coherence pair length (2 context utterances + next): p95 ~160 vibert tokens on train.
MAX_LEN=${MAX_LEN:-192}
# Writes data/dialstart_vi/train${TAG}.pkl. Re-run whenever TOPIC / TOPIC_WINDOW / MAX_LEN change.
python ./data_preprocess.py \
    --datasets dialstart_vi/train \
    --encoder_name NlpHUST/vibert4news-base-cased \
    --topic_encoder_name $TOPIC \
    --save_name "$TAG" \
    --topic_window $TOPIC_WINDOW --topic_max_len $TOPIC_MAX_LEN \
    --max_len $MAX_LEN "$@"
