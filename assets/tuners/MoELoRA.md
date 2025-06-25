# MoELoRA Tuning (MoELoRA)

## Introduction

MoELoRA is a parameter-efficient fine-tuning framework that combines the Mixture of Experts (MoE) mechanism with the Low-Rank Adaptation (LoRA) mechanism, which aims to improve the adaptability of multimodal large language models in dynamic task environments. MoELoRA adaptively allocates expert resources and dynamically adjusts the adaptation weights of different modalities, significantly improving the model's processing effect on multiple tasks while reducing the amount of trainable parameters.

## Parameters

- `train_type` (str): Need to be set to `moelora`.
- `lora_rank` (int): Rank of the low-rank matrix for parameter adaptation, controlling the dimensionality of the incremental updates (default: 8).
- `lora_alpha` (int): Scaling coefficient for the low-rank matrices, adjusting the magnitude of adaptation (default: 32).
- `lora_dropout` (float): Dropout rate applied to the adapter layers to prevent overfitting (default: 0.05).
- `lora_bias` (Literal['none', 'all']): Whether to include bias terms in the adapter layers: `none` disables bias, `all` enables bias (default: `none`).
- `lora_dtype` (Literal['float16', 'bfloat16', 'float32', None]): Data type for adapter parameters (default: None, inherits the base model's precision).
- `lorap_lr_ratio` (Optional[float]): Learning rate scaling ratio for the adapter relative to the base model (if unspecified, uses the same learning rate) (default: None).
- `expert_num` (int): Number of experts (default: 4).

## Citation

```pascal
@article{chen2024coin,
  title={Coin: A benchmark of continual instruction tuning for multimodel large language models},
  author={Chen, Cheng and Zhu, Junchen and Luo, Xu and Shen, Hengtao and Song, Jingkuan and Gao, Lianli},
  journal={Advances in Neural Information Processing Systems},
  volume={37},
  pages={57817--57840},
  year={2024}
}
```
