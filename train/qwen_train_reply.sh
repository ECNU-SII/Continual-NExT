export PYTHONPATH=$PYTHONPATH:peft

NPROC_PER_NODE=4 \
CUDA_VISIBLE_DEVICES=0,1,2,3 swift sft --torch_dtype bfloat16 \
    --add_version False \
    --deepspeed zero0 \
    --model /mnt/workspace/hqs/ckpt/models/Qwen/Qwen2.5-7b --model_type qwen2_5 --template qwen2_5 \
    --system "你是一个智能助手，主要负责处理与2022年新闻相关的对话。整个对话需围绕政治经济、军事战争、文体科教这三大领域展开，回答要基于2022年新闻事件，提供准确、客观的信息，避免主观臆断和不实内容。" \
    --dataset /mnt/workspace/hqs/datasets/2022.json --split_dataset_ratio 0 --max_length 4096 --attn_impl flash_attn \
    --lora_rank 64 --init_weights True --per_device_train_batch_size 4 --num_train_epochs 20 --learning_rate 1e-4 --gradient_accumulation_steps 1 --eval_steps 50 --output_dir output/qwen2_5_reply/2022 \

NPROC_PER_NODE=4 \
CUDA_VISIBLE_DEVICES=0,1,2,3 swift sft --torch_dtype bfloat16 \
    --add_version False \
    --deepspeed zero0 \
    --model /mnt/workspace/hqs/ckpt/models/Qwen/Qwen2.5-7b --model_type qwen2_5 --template qwen2_5 \
    --system "你是一个智能助手，主要负责处理与2022-2023年新闻相关的对话。整个对话需围绕政治经济、军事战争、文体科教这三大领域展开，回答要基于2022-2023年新闻事件，提供准确、客观的信息，避免主观臆断和不实内容。" \
    --dataset /mnt/workspace/hqs/datasets/2023.json --split_dataset_ratio 0 --max_length 4096 --attn_impl flash_attn \
    --lora_rank 64 --init_weights True --per_device_train_batch_size 4 --num_train_epochs 20 --learning_rate 1e-4 --gradient_accumulation_steps 1 --eval_steps 50 --output_dir output/qwen2_5_reply/2023 \
    --adapters /mnt/workspace/hqs/code1/ms-swift-main/output/qwen2_5_reply/2022/checkpoint-260 \
    --replay_datasets /mnt/workspace/hqs/datasets/2022.json 

NPROC_PER_NODE=4 \
CUDA_VISIBLE_DEVICES=0,1,2,3 swift sft --torch_dtype bfloat16 \
    --add_version False \
    --deepspeed zero0 \
    --model /mnt/workspace/hqs/ckpt/models/Qwen/Qwen2.5-7b --model_type qwen2_5 --template qwen2_5 \
    --system "你是一个智能助手，主要负责处理与2022-2024年新闻相关的对话。整个对话需围绕政治经济、军事战争、文体科教这三大领域展开，回答要基于2022-2024年新闻事件，提供准确、客观的信息，避免主观臆断和不实内容。" \
    --dataset /mnt/workspace/hqs/datasets/2024.json --split_dataset_ratio 0 --max_length 4096 --attn_impl flash_attn \
    --lora_rank 64 --init_weights True --per_device_train_batch_size 4 --num_train_epochs 20 --learning_rate 1e-4 --gradient_accumulation_steps 1 --eval_steps 50 --output_dir output/qwen2_5_reply/2024 \
    --adapters /mnt/workspace/hqs/code1/ms-swift-main/output/qwen2_5_reply/2023/checkpoint-280 \
    --replay_datasets '/mnt/workspace/hqs/datasets/2022.json' '/mnt/workspace/hqs/datasets/2023.json'

NPROC_PER_NODE=4 \
CUDA_VISIBLE_DEVICES=0,1,2,3 swift sft --torch_dtype bfloat16 \
    --add_version False \
    --deepspeed zero0 \
    --model /mnt/workspace/hqs/ckpt/models/Qwen/Qwen2.5-7b --model_type qwen2_5 --template qwen2_5 \
    --system "你是一个智能助手，主要负责处理与2022-2025年新闻相关的对话。整个对话需围绕政治经济、军事战争、文体科教这三大领域展开，回答要基于2022-2025年新闻事件，提供准确、客观的信息，避免主观臆断和不实内容。" \
    --dataset /mnt/workspace/hqs/datasets/2025.json --split_dataset_ratio 0 --max_length 4096 --attn_impl flash_attn \
    --lora_rank 64 --init_weights True --per_device_train_batch_size 4 --num_train_epochs 20 --learning_rate 1e-4 --gradient_accumulation_steps 1 --eval_steps 50 --output_dir output/qwen2_5_reply/2025 \
    --adapters /mnt/workspace/hqs/code1/ms-swift-main/output/qwen2_5_reply/2024/checkpoint-380 \
    --replay_datasets '/mnt/workspace/hqs/datasets/2022.json' '/mnt/workspace/hqs/datasets/2023.json' '/mnt/workspace/hqs/datasets/2024.json'


