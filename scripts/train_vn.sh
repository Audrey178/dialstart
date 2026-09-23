python ./data_preprocess.py \
    --datasets vn_synth_train \
    --encoder_name NlpHUST/vibert4news-base-cased \
    --max_len 160

python ./train.py --dataset vn \
    --save_model_name model_vn \
    --topic_model_name NlpHUST/vibert4news-base-cased \
    --coheren_model_name NlpHUST/vibert4news-base-cased \
    --optim adamw8bit --grad_checkpoint --batch_size 32 --accum 1 --epoch 5 --patience 2 \
    --val_dataset vn_synth_val --eval_max_len 160 --eval_oracle_boundary_count \
    --train_eval_dataset vn_synth_train --train_eval_docs 100
