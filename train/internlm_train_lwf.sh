export PYTHONPATH=$PYTHONPATH:peft

NPROC_PER_NODE=2 \
CUDA_VISIBLE_DEVICES=0,1 swift sft --torch_dtype bfloat16 \
    --add_version False \
    --deepspeed zero0 \
    --model Shanghai_AI_Laboratory/internlm2_5-7b-chat --model_type internlm2 --template internlm2 \
    --system "你是一个智能助手，主要负责处理与2022年新闻相关的对话。整个对话需围绕政治经济、军事战争、文体科教这三大领域展开，回答要基于2022年新闻事件，提供准确、客观的信息，避免主观臆断和不实内容。" \
    --dataset datasets/2022.json --split_dataset_ratio 0 --max_length 4096 --train_dataloader_shuffle False --dataloader_num_workers 4 --attn_impl flash_attn \
    --lora_rank 64 --init_weights True --per_device_train_batch_size 4 --num_train_epochs 20 --learning_rate 1e-4 --gradient_accumulation_steps 1 --eval_steps 50 --output_dir output/internlm2_5_lwf/2022 \
    --use_lwf False

NPROC_PER_NODE=2 \
CUDA_VISIBLE_DEVICES=0,1 swift sft --torch_dtype bfloat16 \
    --add_version False \
    --deepspeed zero0 \
    --model Shanghai_AI_Laboratory/internlm2_5-7b-chat --model_type internlm2 --template internlm2 \
    --system "你是一个智能助手，主要负责处理与2022-2023年新闻相关的对话。整个对话需围绕政治经济、军事战争、文体科教这三大领域展开，回答要基于2022-2023年新闻事件，提供准确、客观的信息，避免主观臆断和不实内容。" \
    --dataset datasets/2023.json --split_dataset_ratio 0 --max_length 4096 --train_dataloader_shuffle False --dataloader_num_workers 4 --attn_impl flash_attn \
    --lora_rank 64 --init_weights True --per_device_train_batch_size 4 --num_train_epochs 20 --learning_rate 1e-4 --gradient_accumulation_steps 1 --eval_steps 50 --output_dir output/internlm2_5_lwf/2023 \
    --adapters ms-swift-main/output/internlm2_5_lwf/2022/checkpoint-520 \
    --use_lwf True

NPROC_PER_NODE=2 \
CUDA_VISIBLE_DEVICES=0,1 swift sft --torch_dtype bfloat16 \
    --add_version False \
    --deepspeed zero0 \
    --model Shanghai_AI_Laboratory/internlm2_5-7b-chat --model_type internlm2 --template internlm2 \
    --system "你是一个智能助手，主要负责处理与2022-2024年新闻相关的对话。整个对话需围绕政治经济、军事战争、文体科教这三大领域展开，回答要基于2022-2024年新闻事件，提供准确、客观的信息，避免主观臆断和不实内容。" \
    --dataset datasets/2024.json --split_dataset_ratio 0 --max_length 4096 --train_dataloader_shuffle False --dataloader_num_workers 4 --attn_impl flash_attn \
    --lora_rank 64 --init_weights True --per_device_train_batch_size 4 --num_train_epochs 20 --learning_rate 1e-4 --gradient_accumulation_steps 1 --eval_steps 50 --output_dir output/internlm2_5_lwf/2024 \
    --adapters ms-swift-main/output/internlm2_5_lwf/2023/checkpoint-500 \
    --use_lwf True

NPROC_PER_NODE=2 \
CUDA_VISIBLE_DEVICES=0,1 swift sft --torch_dtype bfloat16 \
    --add_version False \
    --deepspeed zero0 \
    --model Shanghai_AI_Laboratory/internlm2_5-7b-chat --model_type internlm2 --template internlm2 \
    --system "你是一个智能助手，主要负责处理与2022-2025年新闻相关的对话。整个对话需围绕政治经济、军事战争、文体科教这三大领域展开，回答要基于2022-2025年新闻事件，提供准确、客观的信息，避免主观臆断和不实内容。" \
    --dataset datasets/2025.json --split_dataset_ratio 0 --max_length 4096 --train_dataloader_shuffle False --dataloader_num_workers 4 --attn_impl flash_attn \
    --lora_rank 64 --init_weights True --per_device_train_batch_size 4 --num_train_epochs 20 --learning_rate 1e-4 --gradient_accumulation_steps 1 --eval_steps 50 --output_dir output/internlm2_5_lwf/2025 \
    --adapters ms-swift-main/output/internlm2_5_lwf/2024/checkpoint-660 \
    --use_lwf True
    


