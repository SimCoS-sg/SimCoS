NUM_GPUS=4

nohup torchrun --standalone --nnodes=1 --nproc_per_node=$NUM_GPUS main.py \
    --model_name_or_path=./tmp/outgen_mymodel_alpha_1\
    --do_predict \
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
    --num_train_epochs=1 \
    --logging_steps=1 \
    --learning_rate=3e-5 \
    --warmup_steps=100 \
    --load_best_model_at_end \
    --ignore_data_skip \
    --evaluation_strategy steps \
    --predict_with_generate \
    --gradient_accumulation_steps=40 > output.log 2>&1 &