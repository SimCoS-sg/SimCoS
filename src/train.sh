NUM_GPUS=4
nohup torchrun --standalone --nnodes=1 --nproc_per_node=$NUM_GPUS main.py \
    --model_name_or_path=./model \
    --do_train \
    --do_eval \
    --train_file ./data/gen_train.json \
    --validation_file ./data/gen_test.json \
    --test_file ./data/gen_test.json \
    --output_dir=./tmp/outgen_mymodel_alpha_1 \
    --overwrite_output_dir \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --text_column src \
    --summary_column tgt \
    --save_total_limit=10 \
    --num_train_epochs=50 \
    --logging_steps=1 \
    --learning_rate=1e-4 \
    --warmup_steps=100 \
    --ignore_data_skip \
    --evaluation_strategy steps \
    --eval_steps 20\
    --predict_with_generate \
    --gradient_accumulation_steps=20 > output.log 2>&1 &
# NUM_GPUS=1
#     # --do_eval \
# torchrun --standalone --nnodes=1 --nproc_per_node=$NUM_GPUS main.py \
#     --model_name_or_path=./model \
#     --do_train \
#     --train_file ./data/gen_train.json \
#     --validation_file ./data/gen_test.json \
#     --test_file ./data/gen_test.json \
#     --output_dir=./tmp/outgen_mymodel_0 \
#     --overwrite_output_dir \
#     --per_device_train_batch_size=4 \
#     --per_device_eval_batch_size=4 \
#     --text_column src \
#     --summary_column tgt \
#     --save_total_limit=10 \
#     --num_train_epochs=20 \
#     --logging_steps=1 \
#     --learning_rate=3e-5 \
#     --warmup_steps=100 \
#     --load_best_model_at_end \
#     --evaluation_strategy steps \
#     --predict_with_generate \
#     --overwrite_cache \
#     --gradient_accumulation_steps=40 