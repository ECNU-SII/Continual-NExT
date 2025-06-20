# AdaLoRA Tuning (AdaLoRA)

## Introduction

AdaLoRA is an efficient parameter fine-tuning method based on adaptive budget allocation. It aims to optimize resource allocation by dynamically adjusting the rank of low-rank matrices, thereby significantly reducing the amount of trainable parameters while maintaining model performance. Its core mechanism is to combine singular value decomposition (SVD) and importance scoring to assign different ranks to different weight matrices: higher ranks are assigned to matrices with high importance to retain more parameter capacity, and the ranks of unimportant matrices are reduced or even frozen to save resources. AdaLoRA regularly evaluates parameter importance through the RankAllocator algorithm and constrains matrix orthogonality to improve training stability. Compared with fixed-rank LoRA, AdaLoRA achieves higher parameter efficiency and stronger generalization capabilities in tasks such as natural language processing.

## Parameters

- `train_type` (str): Need to be set to `adalora`.
- `adalora_target_r` (int): Target rank for adaptive rank allocation, representing the maximum rank limit during training (default: 8).
- `adalora_init_r` (int): Initial rank assigned to low-rank matrices at the beginning of training (default: 12).
- `adalora_tinit` (int): Starting time step (epoch) for adaptive rank updates (default: 0).
- `adalora_tfinal` (int): Ending time step (epoch) for adaptive rank updates (default: 0).
- `adalora_deltaT` (int): Interval (in epochs) between consecutive rank update steps (default: 1).
- `adalora_beta1` (float): Decay rate for importance score estimation in rank allocation (default: 0.85).
- `adalora_beta2` (float): Decay rate for normalization in rank allocation (default: 0.85).
- `adalora_orth_reg_weight` (float): Weight coefficient for orthogonal regularization to stabilize training (default: 0.5).

## Citation

```pascal
@article{zhang2023adalora,
  title={Adalora: Adaptive budget allocation for parameter-efficient fine-tuning},
  author={Zhang, Qingru and Chen, Minshuo and Bukharin, Alexander and Karampatziakis, Nikos and He, Pengcheng and Cheng, Yu and Chen, Weizhu and Zhao, Tuo},
  journal={arXiv preprint arXiv:2303.10512},
  year={2023}
}
```
