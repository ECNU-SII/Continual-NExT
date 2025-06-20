# Adapter Tuning (Adapter)

## Introduction

Adapter Tuning is an efficient parameter fine-tuning method that aims to achieve efficient transfer learning in natural language processing (NLP) tasks by inserting small neural network modules (adapters) between the layers of pre-trained language models. These lightweight modules contain only a small number of trainable parameters and can flexibly adapt to the needs of downstream tasks while retaining most of the fixed parameters of the original model, thereby significantly reducing computing resources and storage overhead. Adapter Tuning has shown comparable performance to full parameter fine-tuning in multiple NLP tasks, while having higher parameter efficiency and deployment flexibility, and has become an important technical solution for low-resource, multi-task, and continuous learning scenarios in recent years.

## Parameters

- `train_type` (str): Need to be set to `adapter`.
- `adapter_act` (str): The type of activation function used in the adapter (default: `gelu`).
- `adapter_length` (int): The number of hidden units in the middle layer of the adapter (default: 128).

## Citation

```pascal
@inproceedings{houlsby2019parameter,
  title={Parameter-efficient transfer learning for NLP},
  author={Houlsby, Neil and Giurgiu, Andrei and Jastrzebski, Stanislaw and Morrone, Bruna and De Laroussilhe, Quentin and Gesmundo, Andrea and Attariyan, Mona and Gelly, Sylvain},
  booktitle={International conference on machine learning},
  pages={2790--2799},
  year={2019},
  organization={PMLR}
}
```
