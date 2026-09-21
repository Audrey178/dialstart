# pick the best epoch checkpoint on val first, then run test once with --single_ckpt
python test.py --model model_vn --dataset vn_val \
    --topic_model_name NlpHUST/vibert4news-base-cased \
    --coheren_model_name NlpHUST/vibert4news-base-cased \
    --max_len 160 --oracle_boundary_count

python test.py --model model_vn --dataset vn_test \
    --topic_model_name NlpHUST/vibert4news-base-cased \
    --coheren_model_name NlpHUST/vibert4news-base-cased \
    --max_len 160 --oracle_boundary_count
