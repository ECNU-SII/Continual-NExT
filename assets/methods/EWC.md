# Elastic Weight Consolidation (EWC)

## Introduction

Elastic Weight Consolidation (EWC) introduces a regularization term based on the Fisher Information Matrix (FIM) to protect model parameters that are critical to the previous task from being significantly modified. Specifically, EWC calculates the importance of each parameter in the old task (the diagonal elements of FIM) and imposes constraints on these important parameters when training new tasks to limit their variation, thereby optimizing the performance of the new task while retaining the knowledge of the old task.

## Parameters

- `EWC` (bool): Whether to enable EWC during training (default: False).
- `EWC_lambda` (float): The weight of the EWC (default: 0.5).
- `EWC_limit` (int): EWC calculates FIM using the upper limit of the samples (default: 1000).
- `EWC_path` (str): Folder path for storing FIM and old parameter backups (default: None).

## Citation

```pascal
@article{kirkpatrick2017overcoming,
    title={Overcoming catastrophic forgetting in neural networks},
    author={Kirkpatrick, James and Pascanu, Razvan and Rabinowitz, Neil and Veness, Joel and Desjardins, Guillaume and Rusu, Andrei A and Milan, Kieran and Quan, John and Ramalho, Tiago and Grabska-Barwinska, Agnieszka and others},
    journal={Proceedings of the national academy of sciences},
    volume={114},
    number={13},
    pages={3521--3526},
    year={2017},
    publisher={National Academy of Sciences}
}
```
