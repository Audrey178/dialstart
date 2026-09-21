python ./data_preprocess.py \
    --datasets vn_synth_train \
    --encoder_name NlpHUST/vibert4news-base-cased \
    --max_len 160

python ./train.py --dataset vn \
    --save_model_name model_vn \
    --topic_model_name NlpHUST/vibert4news-base-cased \
    --coheren_model_name NlpHUST/vibert4news-base-cased \
    --optim adamw8bit --grad_checkpoint --batch_size 1 --accum 12 --epoch 9
