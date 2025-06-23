
CUDA_VISIBLE_DEVICES=1 swift deploy \
    --model internlm2_5-7b-chat \
    --adapters ms-swift-main/output/internlm2_5_test/2023/checkpoint-1000 \
    --max_new_tokens 512 \
    --infer_backend vllm \
    --port 9001 \
    --vllm_max_lora_rank 64



