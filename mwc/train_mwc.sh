NPROC_PER_NODE=2 \
CUDA_VISIBLE_DEVICES=0,1  swift sft --torch_dtype 'bfloat16' \
    --add_version False \
    --deepspeed zero1 \
    --model /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/public/hqs/internlm2_5-7b-chat \
    --model_type 'internlm2' \
    --template 'internlm2' \
    --system "你是一个智能助手，主要负责处理与 2022-2025 年新闻相关的对话。整个对话需围绕政治经济、军事战争、文体科教这三大领域展开，回答要基于 2022-2025 年新闻事件，提供准确、客观的信息，避免主观臆断和不实内容。" \
    --dataset '/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/mengweicheng-240108120092/swift/datasets/train/2022.json' \
    --split_dataset_ratio 0 \
    --max_length '4096' \
    --lora_rank '64' \
    --init_weights 'True' \
    --num_train_epochs 1 \
    --per_device_train_batch_size '4' \
    --learning_rate '1e-4' \
    --attn_impl 'flash_attn' \
    --gradient_accumulation_steps '1' \
    --eval_steps '50' \
    --output_dir '/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/public/hqs/ms-swift-main/mwc/output/internlm2/ewc/2022' \
    --EWC True \
    --EWC_lambda 0.5 \
    --EWC_limit 200



NPROC_PER_NODE=2 \
CUDA_VISIBLE_DEVICES=0,1  swift sft --torch_dtype 'bfloat16' \
    --deepspeed zero1 \
    --model /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/public/hqs/internlm2_5-7b-chat \
    --model_type 'internlm2' \
    --template 'internlm2' \
    --system "你是一个智能助手，主要负责处理与 2022-2025 年新闻相关的对话。整个对话需围绕政治经济、军事战争、文体科教这三大领域展开，回答要基于 2022-2025 年新闻事件，提供准确、客观的信息，避免主观臆断和不实内容。" \
    --dataset '/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/mengweicheng-240108120092/swift/datasets/train/2023.json' \
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
    --output_dir '/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/public/hqs/ms-swift-main/mwc/output/internlm2/ewc/2023' \
    --adapters /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/public/hqs/ms-swift-main/mwc/output/internlm2/ewc/2022/checkpoint-26 \
    --EWC True \
    --EWC_lambda 0.5 \
    --EWC_limit 200 \
    --EWC_path /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/public/hqs/ms-swift-main/mwc/output/internlm2/ewc/2022


NPROC_PER_NODE=2 \
CUDA_VISIBLE_DEVICES=0,1  swift sft --torch_dtype 'bfloat16' \
    --model /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/public/hqs/internlm2_5-7b-chat \
    --deepspeed zero1 \
    --model_type 'internlm2' \
    --template 'internlm2' \
    --system "你是一个智能助手，主要负责处理与 2022-2025 年新闻相关的对话。整个对话需围绕政治经济、军事战争、文体科教这三大领域展开，回答要基于 2022-2025 年新闻事件，提供准确、客观的信息，避免主观臆断和不实内容。" \
    --dataset '/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/mengweicheng-240108120092/swift/datasets/train/2024.json' \
    --split_dataset_ratio 0 \
    --max_length '4096' \
    --lora_rank '64' \
    --init_weights 'True' \
    --num_train_epochs 1 \
    --per_device_train_batch_size '4' \
    --learning_rate '1e-4' \
    --attn_impl 'flash_attn' \
    --gradient_accumulation_steps '1' \
    --eval_steps '50' \
    --output_dir '/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/public/hqs/ms-swift-main/mwc/output/internlm2/ewc/2024' \
    --adapters /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/public/hqs/ms-swift-main/mwc/output/internlm2/ewc/2023/model/checkpoint-1000 \
    --EWC True \
    --EWC_lambda 0.5 \
    --EWC_limit 200 \
    --EWC_path /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/public/hqs/ms-swift-main/mwc/output/internlm2/ewc/2023/model

NPROC_PER_NODE=2 \
CUDA_VISIBLE_DEVICES=0,1  swift sft --torch_dtype 'bfloat16' \
    --model /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/public/hqs/internlm2_5-7b-chat \
    --deepspeed zero1 \
    --model_type 'internlm2' \
    --template 'internlm2' \
    --system "你是一个智能助手，主要负责处理与 2022-2025 年新闻相关的对话。整个对话需围绕政治经济、军事战争、文体科教这三大领域展开，回答要基于 2022-2025 年新闻事件，提供准确、客观的信息，避免主观臆断和不实内容。" \
    --dataset '/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/mengweicheng-240108120092/swift/datasets/train/2025.json' \
    --split_dataset_ratio 0 \
    --max_length '4096' \
    --lora_rank '64' \
    --init_weights 'True' \
    --num_train_epochs 1 \
    --per_device_train_batch_size '4' \
    --learning_rate '1e-4' \
    --attn_impl 'flash_attn' \
    --gradient_accumulation_steps '1' \
    --eval_steps '50' \
    --output_dir '/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/public/hqs/ms-swift-main/mwc/output/internlm2/ewc/2025' \
    --adapters /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/public/hqs/ms-swift-main/mwc/output/internlm2/ewc/2024/model/checkpoint-1320 \
    --EWC True \
    --EWC_lambda 0.5 \
    --EWC_limit 200 \
    --EWC_path /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/public/hqs/ms-swift-main/mwc/output/internlm2/ewc/2024/model


NPROC_PER_NODE=2 \
CUDA_VISIBLE_DEVICES=0,1  swift sft --torch_dtype 'bfloat16' \
    --model /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/public/hqs/internlm2_5-7b-chat \
    --deepspeed zero0 \
    --model_type 'internlm2' \
    --template 'internlm2' \
    --system "你是一个智能助手，主要负责处理与 2022-2025 年新闻相关的对话。整个对话需围绕政治经济、军事战争、文体科教这三大领域展开，回答要基于 2022-2025 年新闻事件，提供准确、客观的信息，避免主观臆断和不实内容。" \
    --dataset '/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/mengweicheng-240108120092/swift/datasets/train/2023.json' \
    --split_dataset_ratio 0 \
    --max_length '4096' \
    --lora_rank '64' \
    --init_weights 'True' \
    --num_train_epochs 1 \
    --per_device_train_batch_size '4' \
    --learning_rate '1e-4' \
    --attn_impl 'flash_attn' \
    --gradient_accumulation_steps '1' \
    --eval_steps '50' \
    --output_dir '/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/public/hqs/ms-swift-main/mwc/output/internlm2/gem/2023' \
    --adapters /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/public/hqs/ms-swift-main/mwc/output/internlm2/ewc/2022/checkpoint-26 \
    --GEM True \
    --GEM_memory_strength 0.5 \
    --GEM_replay_ratio 0.2 \
    --GEM_replay_task_list '2022.json' \
    --GEM_previous_task_dataset /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/mengweicheng-240108120092/swift/datasets/train/


NPROC_PER_NODE=2 \
CUDA_VISIBLE_DEVICES=0,1  swift sft --torch_dtype 'bfloat16' \
    --model /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/public/hqs/internlm2_5-7b-chat \
    --deepspeed zero1 \
    --model_type 'internlm2' \
    --template 'internlm2' \
    --system "你是一个智能助手，主要负责处理与 2022-2025 年新闻相关的对话。整个对话需围绕政治经济、军事战争、文体科教这三大领域展开，回答要基于 2022-2025 年新闻事件，提供准确、客观的信息，避免主观臆断和不实内容。" \
    --dataset '/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/mengweicheng-240108120092/swift/datasets/train/2024.json' \
    --split_dataset_ratio 0 \
    --max_length '4096' \
    --lora_rank '64' \
    --init_weights 'True' \
    --num_train_epochs 1 \
    --per_device_train_batch_size '4' \
    --learning_rate '1e-4' \
    --attn_impl 'flash_attn' \
    --gradient_accumulation_steps '1' \
    --eval_steps '50' \
    --output_dir '/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/public/hqs/ms-swift-main/mwc/output/internlm2/gem/2024' \
    --adapters /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/public/hqs/ms-swift-main/mwc/output/internlm2/gem/2023/model/checkpoint-1200 \
    --GEM True \
    --GEM_memory_strength 0.5 \
    --GEM_replay_ratio 0.2 \
    --GEM_replay_task_list '2022.json,2023.json' \
    --GEM_previous_task_dataset /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/mengweicheng-240108120092/swift/datasets/train/


NPROC_PER_NODE=2 \
CUDA_VISIBLE_DEVICES=0,1  swift sft --torch_dtype 'bfloat16' \
    --model /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/public/hqs/internlm2_5-7b-chat \
    --deepspeed zero1 \
    --model_type 'internlm2' \
    --template 'internlm2' \
    --system "你是一个智能助手，主要负责处理与 2022-2025 年新闻相关的对话。整个对话需围绕政治经济、军事战争、文体科教这三大领域展开，回答要基于 2022-2025 年新闻事件，提供准确、客观的信息，避免主观臆断和不实内容。" \
    --dataset '/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/mengweicheng-240108120092/swift/datasets/train/2025.json' \
    --split_dataset_ratio 0 \
    --max_length '4096' \
    --lora_rank '64' \
    --init_weights 'True' \
    --num_train_epochs 1 \
    --per_device_train_batch_size '4' \
    --learning_rate '1e-4' \
    --attn_impl 'flash_attn' \
    --gradient_accumulation_steps '1' \
    --eval_steps '50' \
    --output_dir '/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/public/hqs/ms-swift-main/mwc/output/internlm2/gem/2025' \
    --adapters /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/public/hqs/ms-swift-main/mwc/output/internlm2/gem/2024/model/checkpoint-1720 \
    --GEM True \
    --GEM_memory_strength 0.5 \
    --GEM_replay_ratio 0.2 \
    --GEM_replay_task_list '2022.json,2023.json,2024.json' \
    --GEM_previous_task_dataset /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/mengweicheng-240108120092/swift/datasets/train/



NPROC_PER_NODE=2 \
CUDA_VISIBLE_DEVICES=0,1  swift sft --torch_dtype 'bfloat16' \
    --model /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/public/hqs/Qwen2.5-7b \
    --deepspeed zero1 \
    --model_type 'qwen2_5' \
    --template 'qwen2_5' \
    --system "你是一个智能助手，主要负责处理与 2022-2025 年新闻相关的对话。整个对话需围绕政治经济、军事战争、文体科教这三大领域展开，回答要基于 2022-2025 年新闻事件，提供准确、客观的信息，避免主观臆断和不实内容。" \
    --dataset '/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/mengweicheng-240108120092/swift/datasets/train/2022.json' \
    --split_dataset_ratio 0 \
    --max_length '4096' \
    --lora_rank '64' \
    --init_weights 'True' \
    --num_train_epochs 1 \
    --per_device_train_batch_size '8' \
    --learning_rate '1e-4' \
    --attn_impl 'flash_attn' \
    --gradient_accumulation_steps '1' \
    --eval_steps '50' \
    --output_dir '/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/public/hqs/ms-swift-main/mwc/output/qwen2_5/ewc/2022' \
    --EWC True \
    --EWC_lambda 0.5 \
    --EWC_limit 200

NPROC_PER_NODE=2 \
CUDA_VISIBLE_DEVICES=0,1  swift sft --torch_dtype 'bfloat16' \
    --model /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/public/hqs/Qwen2.5-7b \
    --deepspeed zero1 \
    --model_type 'qwen2_5' \
    --template 'qwen2_5' \
    --system "你是一个智能助手，主要负责处理与 2022-2025 年新闻相关的对话。整个对话需围绕政治经济、军事战争、文体科教这三大领域展开，回答要基于 2022-2025 年新闻事件，提供准确、客观的信息，避免主观臆断和不实内容。" \
    --dataset '/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/mengweicheng-240108120092/swift/datasets/train/2023.json' \
    --split_dataset_ratio 0 \
    --max_length '4096' \
    --lora_rank '64' \
    --init_weights 'True' \
    --num_train_epochs 1 \
    --per_device_train_batch_size '4' \
    --learning_rate '1e-4' \
    --attn_impl 'flash_attn' \
    --gradient_accumulation_steps '1' \
    --eval_steps '50' \
    --output_dir '/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/public/hqs/ms-swift-main/mwc/output/qwen2_5/ewc/2023' \
    --adapters /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/public/hqs/ms-swift-main/mwc/output/qwen2_5/ewc/2022/model/checkpoint-26 \
    --EWC True \
    --EWC_lambda 0.5 \
    --EWC_limit 200 \
    --EWC_path /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/public/hqs/ms-swift-main/mwc/output/qwen2_5/ewc/2022/model

NPROC_PER_NODE=2 \
CUDA_VISIBLE_DEVICES=0,1  swift sft --torch_dtype 'bfloat16' \
    --model /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/public/hqs/Qwen2.5-7b \
    --deepspeed zero1 \
    --model_type 'qwen2_5' \
    --template 'qwen2_5' \
    --system "你是一个智能助手，主要负责处理与 2022-2025 年新闻相关的对话。整个对话需围绕政治经济、军事战争、文体科教这三大领域展开，回答要基于 2022-2025 年新闻事件，提供准确、客观的信息，避免主观臆断和不实内容。" \
    --dataset '/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/mengweicheng-240108120092/swift/datasets/train/2023.json' \
    --split_dataset_ratio 0 \
    --max_length '4096' \
    --lora_rank '64' \
    --init_weights 'True' \
    --num_train_epochs 1 \
    --per_device_train_batch_size '4' \
    --learning_rate '1e-4' \
    --attn_impl 'flash_attn' \
    --gradient_accumulation_steps '1' \
    --eval_steps '50' \
    --output_dir '/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/public/hqs/ms-swift-main/mwc/output/qwen2_5/gem/2023' \
    --adapters /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/public/hqs/ms-swift-main/mwc/output/qwen2_5/ewc/2022/model/checkpoint-26 \
    --GEM True \
    --GEM_memory_strength 0.5 \
    --GEM_replay_ratio 0.2 \
    --GEM_replay_task_list '2022.json' \
    --GEM_previous_task_dataset /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/mengweicheng-240108120092/swift/datasets/train/