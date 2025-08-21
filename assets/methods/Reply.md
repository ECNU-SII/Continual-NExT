# Experience Replay (Replay)

## Introduction

Experience Replay serves as a fundamental continual learning technique, primarily designed to preserve a model’s performance on previous tasks by replaying a selected subset of data from historical tasks during the training of new tasks. This approach effectively mitigates catastrophic forgetting—a phenomenon where models drastically lose previously learned knowledge when acquiring new information—by systematically revisiting and relearning from past experiences. Typically, Experience Replay involves constructing a replay buffer that stores a sample of historical data. During new-task training, the model is fed a mixture of fresh data and replayed data from this buffer. By alternating between learning new and reinforcing old information, the method ensures that the model not only adapts to new tasks but also retains proficiency in historical ones, offering a robust solution for enabling stable continual knowledge retention in machine learning systems.

## Parameters
- `replay_task_list` (str): Comma-separated list of previous tasks to replay from
- `replay_nums` (float): The number of samples to replay from each previous task (default: 20)

## Citation

```pascal
@article{rolnick2019experience,
  title={Experience replay for continual learning},
  author={Rolnick, David and Ahuja, Arun and Schwarz, Jonathan and Lillicrap, Timothy and Wayne, Gregory},
  journal={Advances in neural information processing systems},
  volume={32},
  year={2019}
}
```


