# Large Continual Instruction Assistant (CIA)

## Introduction

Learning Without Forgetting (LWF) stands as a prominent continual learning method, specifically crafted to address the "catastrophic forgetting" problem where models forget previously learned knowledge when acquiring new tasks. This approach ingeniously employs knowledge distillation, leveraging a pre-trained model on prior tasks (termed the "teacher model") to guide the current model in retaining knowledge representations from historical tasks while learning novel ones. Notably, LWF operates without requiring access to original old-task datasets; instead, it ensures performance maintenance on previously learned tasks by constraining the output distribution of the current model to match that of the teacher model. This mechanism not only alleviates the burden of data storage but also provides a pivotal solution for enabling efficient continual knowledge accumulation in machine learning models.

## Parameters

- `use_lwf` (bool): Whether to enable LWF during training.
- `alpha` (float): The basic weight of the LWF weight (default: 0.0001).

## Citation

```pascal
@article{li2017learning,
  title={Learning without forgetting},
  author={Li, Zhizhong and Hoiem, Derek},
  journal={IEEE transactions on pattern analysis and machine intelligence},
  volume={40},
  number={12},
  pages={2935--2947},
  year={2017},
  publisher={IEEE}
}
```

