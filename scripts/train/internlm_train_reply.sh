NPROC_PER_NODE=2 \
CUDA_VISIBLE_DEVICES=0,1 swift sft --torch_dtype bfloat16 \
    --add_version False \
    --deepspeed zero0 \
    --model Shanghai_AI_Laboratory/internlm2_5-7b-chat --model_type internlm2 --template internlm2 \
    --system "你是一个智能助手，主要负责处理与2022年新闻相关的对话。整个对话需围绕政治经济、军事战争、文体科教这三大领域展开，回答要基于2022年新闻事件，提供准确、客观的信息，避免主观臆断和不实内容。" \
    --dataset datasets/2022-train.json --split_dataset_ratio 0 --max_length 4096 --attn_impl flash_attn \
    --lora_rank 64 --init_weights True --per_device_train_batch_size 4 --num_train_epochs 20 --learning_rate 1e-4 --gradient_accumulation_steps 1 --eval_steps 50 --output_dir output/internlm2_5_reply/2022 \

NPROC_PER_NODE=2 \
CUDA_VISIBLE_DEVICES=0,1 swift sft --torch_dtype bfloat16 \
    --add_version False \
    --deepspeed zero0 \
    --model Shanghai_AI_Laboratory/internlm2_5-7b-chat --model_type internlm2 --template internlm2 \
    --system "你是一个智能助手，主要负责处理与2022-2023年新闻相关的对话。整个对话需围绕政治经济、军事战争、文体科教这三大领域展开，回答要基于2022-2023年新闻事件，提供准确、客观的信息，避免主观臆断和不实内容。" \
    --dataset datasets/2023-train.json --split_dataset_ratio 0 --max_length 4096 --attn_impl flash_attn \
    --lora_rank 64 --init_weights True --per_device_train_batch_size 4 --num_train_epochs 20 --learning_rate 1e-4 --gradient_accumulation_steps 1 --eval_steps 50 --output_dir output/internlm2_5_reply/2023 \
    --resume_from_checkpoint ./output/internlm2_5_reply/2022/checkpoint-600 \
    --resume_only_model True \
    --replay_datasets 'datasets/2022-train.json '

NPROC_PER_NODE=2 \
CUDA_VISIBLE_DEVICES=0,1 swift sft --torch_dtype bfloat16 \
    --add_version False \
    --deepspeed zero0 \
    --model Shanghai_AI_Laboratory/internlm2_5-7b-chat --model_type internlm2 --template internlm2 \
    --system "你是一个智能助手，主要负责处理与2022-2024年新闻相关的对话。整个对话需围绕政治经济、军事战争、文体科教这三大领域展开，回答要基于2022-2024年新闻事件，提供准确、客观的信息，避免主观臆断和不实内容。" \
    --dataset datasets/2024-train.json --split_dataset_ratio 0 --max_length 4096 --attn_impl flash_attn \
    --lora_rank 64 --init_weights True --per_device_train_batch_size 4 --num_train_epochs 20 --learning_rate 1e-4 --gradient_accumulation_steps 1 --eval_steps 50 --output_dir output/internlm2_5_reply/2024 \
    --resume_from_checkpoint ./output/internlm2_5_reply/2023/checkpoint-600 \
    --resume_only_model True \
    --replay_datasets 'datasets/2022-train.json' 'datasets/2023-train.json'

NPROC_PER_NODE=2 \
CUDA_VISIBLE_DEVICES=0,1 swift sft --torch_dtype bfloat16 \
    --add_version False \
    --deepspeed zero0 \
    --model Shanghai_AI_Laboratory/internlm2_5-7b-chat --model_type internlm2 --template internlm2 \
    --system "你是一个智能助手，主要负责处理与2022-2025年新闻相关的对话。整个对话需围绕政治经济、军事战争、文体科教这三大领域展开，回答要基于2022-2025年新闻事件，提供准确、客观的信息，避免主观臆断和不实内容。" \
    --dataset datasets/2025-train.json --split_dataset_ratio 0 --max_length 4096 --attn_impl flash_attn \
    --lora_rank 64 --init_weights True --per_device_train_batch_size 4 --num_train_epochs 20 --learning_rate 1e-4 --gradient_accumulation_steps 1 --eval_steps 50 --output_dir output/internlm2_5_reply/2025 \
    --resume_from_checkpoint ./output/internlm2_5_reply/2024/checkpoint-760 \
    --resume_only_model True \
    --replay_datasets 'datasets/2022-train.json' 'datasets/2023-train.json' 'datasets/2024-train.json'


