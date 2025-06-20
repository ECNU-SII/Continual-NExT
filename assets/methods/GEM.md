# Gradient Episodic Memory (GEM)

## Introduction

Gradient Episodic Memory (GEM) prevents the model from forgetting old knowledge when learning new tasks by maintaining episodic memory (storing representative samples of historical tasks) and gradient constraints. Its core mechanism is to ensure that the parameter update of the new task does not increase the loss function value of the old task by projecting the gradient of the current task to a subspace that satisfies the gradient direction of all historical tasks. Specifically, GEM optimizes the projected gradient through quadratic programming, so that the model can maintain the stability of the performance of old tasks while minimizing the loss of the current task.

## Parameters

- `GEM` (bool): Whether to enable the GEM during training (default: False).
- `GEM_memory_strength` (float): The strength of the GEM projection constraint (default: 0.5).
- `GEM_replay_ratio` (float): The fraction of samples retained from each past task (default: 0.2).
- `GEM_replay_task_list` (str): Comma separated list of task names to remember (default: None).
- `GEM_maxsamples_list` (str): Comma-separated list of maximum number of samples to keep per memory task (default: None).
- `GEM_previous_task_dataset` (str): Path to the directory containing the datasets for previous tasks (default: None).


## Citation

```pascal
@article{lopez2017gradient,
  title={Gradient episodic memory for continual learning},
  author={Lopez-Paz, David and Ranzato, Marc'Aurelio},
  journal={Advances in neural information processing systems},
  volume={30},
  year={2017}
}
```
