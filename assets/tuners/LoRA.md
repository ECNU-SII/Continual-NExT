# LoRA Tuning (LoRA)

## Introduction

LoRA is a parameter-efficient fine-tuning technology that aims to significantly reduce the amount of trainable parameters by introducing a low-rank decomposition matrix into the weight matrix of a pre-trained model. The core idea is to freeze the weights of the original model and use the low-rank matrix to capture the incremental information related to the task, which not only retains the general capabilities of the pre-trained model but also reduces the computational and storage costs. LoRA is widely used in fields such as natural language processing and computer vision. It has become one of the mainstream methods for fine-tuning large models due to its low memory usage, fast training speed, and support for flexible switching of multiple tasks.

## Parameters

- `train_type` (str): Need to be set to `lora`.
- `lora_rank` (int): Rank of the low-rank matrix for parameter adaptation, controlling the dimensionality of the incremental updates (default: 8).
- `lora_alpha` (int): Scaling coefficient for the low-rank matrices, adjusting the magnitude of adaptation (default: 32).
- `lora_dropout` (float): Dropout rate applied to the adapter layers to prevent overfitting (default: 0.05).
- `lora_bias` (Literal['none', 'all']): Whether to include bias terms in the adapter layers: `none` disables bias, `all` enables bias (default: `none`).
- `lora_dtype` (Literal['float16', 'bfloat16', 'float32', None]): Data type for adapter parameters (default: None, inherits the base model's precision).
- `lorap_lr_ratio` (Optional[float]): Learning rate scaling ratio for the adapter relative to the base model (if unspecified, uses the same learning rate) (default: None).
- `use_rslora` (bool): Whether to enable RS-LoRA (Rank-Stabilized LoRA) for dynamic rank distribution and improved stability (default: False).
- `use_dora` (bool): Whether to enable DoRA (Weight-Decomposed LoRA) for enhanced adaptation via weight decomposition (default: False).

## Citation

```pascal
@article{hu2022lora,
  title={Lora: Low-rank adaptation of large language models.},
  author={Hu, Edward J and Shen, Yelong and Wallis, Phillip and Allen-Zhu, Zeyuan and Li, Yuanzhi and Wang, Shean and Wang, Lu and Chen, Weizhu and others},
  journal={ICLR},
  volume={1},
  number={2},
  pages={3},
  year={2022}
}
```
