# Large Continual Instruction Assistant (CIA)

## Introduction

Starting from the trade-off prerequisite and EMA update and the plasticity and stability ideal condition, CIA generates  the optimal balance weight can be automatically determined by the gradients and learned parameters. Based on the semantic similarity of the instructions, CIA can determine whether to retrain or expand the training parameters and allocate the most suitable parameters for the testing instances. 

## Parameters

- `use_cia` (bool): Whether to enable CIA during training.
- `base_weight` (float): The basic weight of the EMA weight (default: 0.001).

## Citation

```pascal
@inproceedings{qiao2025LCIA,
  title={Large Continual Instruction Assistant},
  author={Qiao, Jingyang and Zhang, Zhizhong and Tan, Xin and Qu, Yanyun and Ding, Shouhong and Xie, Yuan},
  booktitle={International conference on machine learning},
  year={2025}
}
```

