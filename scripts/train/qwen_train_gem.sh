NPROC_PER_NODE=2 \
CUDA_VISIBLE_DEVICES=0,1  swift sft --torch_dtype 'bfloat16' \
    --model ../Qwen2.5-7b \
    --add_version False \
    --deepspeed zero0 \
    --model_type 'qwen2_5' \
    --template 'qwen2_5' \
    --system "你是一个智能助手，主要负责处理与 2022-2025 年新闻相关的对话。整个对话需围绕政治经济、军事战争、文体科教这三大领域展开，回答要基于 2022-2025 年新闻事件，提供准确、客观的信息，避免主观臆断和不实内容。" \
    --dataset '../datasets/train/2022-train.json' \
    --split_dataset_ratio 0 \
    --max_length '4096' \
    --lora_rank '64' \
    --init_weights 'True' \
    --num_train_epochs 20 \
    --per_device_train_batch_size '4' \
    --learning_rate '1e-4' \
    --attn_impl 'flash_attn' \
    --gradient_accumulation_steps '1' \
    --eval_steps '50' \
    --output_dir './output/qwen2_5/gem/2022'


NPROC_PER_NODE=2 \
CUDA_VISIBLE_DEVICES=0,1  swift sft --torch_dtype 'bfloat16' \
    --model ../Qwen2.5-7b \
    --add_version False \
    --deepspeed zero0 \
    --model_type 'qwen2_5' \
    --template 'qwen2_5' \
    --system "你是一个智能助手，主要负责处理与 2022-2025 年新闻相关的对话。整个对话需围绕政治经济、军事战争、文体科教这三大领域展开，回答要基于 2022-2025 年新闻事件，提供准确、客观的信息，避免主观臆断和不实内容。" \
    --dataset '../datasets/train/2023-train.json' \
    --split_dataset_ratio 0 \
    --max_length '4096' \
    --lora_rank '64' \
    --init_weights 'True' \
    --num_train_epochs 20 \
    --per_device_train_batch_size '4' \
    --learning_rate '1e-4' \
    --attn_impl 'flash_attn' \
    --gradient_accumulation_steps '1' \
    --eval_steps '50' \
    --output_dir './output/qwen2_5/gem/2023' \
    --resume_from_checkpoint ./output/qwen2_5/gem/2022/checkpoint-560 \
    --resume_only_model True \
    --GEM True \
    --GEM_memory_strength 0.5 \
    --GEM_replay_ratio 0.2 \
    --GEM_replay_task_list '2022-train.json' \
    --GEM_previous_task_dataset ../datasets/train/


NPROC_PER_NODE=2 \
CUDA_VISIBLE_DEVICES=0,1  swift sft --torch_dtype 'bfloat16' \
    --model ../Qwen2.5-7b \
    --add_version False \
    --deepspeed zero0 \
    --model_type 'qwen2_5' \
    --template 'qwen2_5' \
    --system "你是一个智能助手，主要负责处理与 2022-2025 年新闻相关的对话。整个对话需围绕政治经济、军事战争、文体科教这三大领域展开，回答要基于 2022-2025 年新闻事件，提供准确、客观的信息，避免主观臆断和不实内容。" \
    --dataset '../datasets/train/2024-train.json' \
    --split_dataset_ratio 0 \
    --max_length '4096' \
    --lora_rank '64' \
    --init_weights 'True' \
    --num_train_epochs 20 \
    --per_device_train_batch_size '4' \
    --learning_rate '1e-4' \
    --attn_impl 'flash_attn' \
    --gradient_accumulation_steps '1' \
    --eval_steps '50' \
    --output_dir './output/qwen2_5/gem/2024' \
    --resume_from_checkpoint ./output/qwen2_5/gem/2023/checkpoint-680 \
    --resume_only_model True \
    --GEM True \
    --GEM_memory_strength 0.5 \
    --GEM_replay_ratio 0.2 \
    --GEM_replay_task_list '2022-train.json,2023-train.json' \
    --GEM_previous_task_dataset ../datasets/train/


NPROC_PER_NODE=2 \
CUDA_VISIBLE_DEVICES=0,1  swift sft --torch_dtype 'bfloat16' \
    --model ../Qwen2.5-7b \
    --add_version False \
    --deepspeed zero0 \
    --model_type 'qwen2_5' \
    --template 'qwen2_5' \
    --system "你是一个智能助手，主要负责处理与 2022-2025 年新闻相关的对话。整个对话需围绕政治经济、军事战争、文体科教这三大领域展开，回答要基于 2022-2025 年新闻事件，提供准确、客观的信息，避免主观臆断和不实内容。" \
    --dataset '../datasets/train/2025-train.json' \
    --split_dataset_ratio 0 \
    --max_length '4096' \
    --lora_rank '64' \
    --init_weights 'True' \
    --num_train_epochs 20 \
    --per_device_train_batch_size '4' \
    --learning_rate '1e-4' \
    --attn_impl 'flash_attn' \
    --gradient_accumulation_steps '1' \
    --eval_steps '50' \
    --output_dir './output/qwen2_5/gem/2025' \
    --resume_from_checkpoint ./output/qwen2_5/gem/2024/checkpoint-940 \
    --resume_only_model True \
    --GEM True \
    --GEM_memory_strength 0.5 \
    --GEM_replay_ratio 0.2 \
    --GEM_replay_task_list '2022-train.json,2023-train.json,2024-train.json' \
    --GEM_previous_task_dataset ../datasets/train/

