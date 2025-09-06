![# Continual-NExT](assets/ecnu-sii.jpg)

[![GitHub Repo stars](https://img.shields.io/github/stars/ecnu-sii/Continual-NExT?style=social)](https://github.com/ecnu-sii/Continual-NExT/stargazers)
[![GitHub last commit](https://img.shields.io/github/last-commit/ecnu-sii/Continual-NExT)](https://github.com/ecnu-sii/Continual-NExT/commits/main)
[![GitHub contributors](https://img.shields.io/github/contributors/ecnu-sii/Continual-NExT?color=orange)](https://github.com/ecnu-sii/Continual-NExT/graphs/contributors)
[![News Datasets](https://img.shields.io/badge/🤗-News%20Datasets-blue)](https://huggingface.co/datasets/ECNU-SII/Continual-News)
[![NexT Datasets](https://img.shields.io/badge/🤗-NexT%20Datasets-blue)](https://huggingface.co/datasets/ECNU-SII/Continual-NexT)

## Introduction

🔥 Continual-NExT is a continual learning toolkit and benchmark for **Large Foundation Modals (LFMs)** developed based on the ms-swift framework, focusing on the catastrophic forgetting of LFMs in the process of continual evolution. It integrates multiple modalities, models, tuning paradigms, and continual learning (CL) methods, allowing researchers to freely combine these components in developing and testing new methods to solve **the trade-off between stability and plasticity** in LFMs.

⚙️ In addition to using the datasets supported by the ms-swift framework, Continual-NExT also supports interfaces for continual fine-tuning of public and private datasets, the formation of annotation json please kindly refer to **Supported Dataset Formats**. Specifically, we provide a new open-source dataset based on Large Language Models (LLMs), **Continual-News Knowledge Evolution** to help researchers better understand the continual evolution process of LLMs. A longest known multimodal continual instruction tuning benchmark: **Continual-NExT** is proposed for further validation of the continual learning ability in multimodal instruction following.

📄 In summary, our toolkit and benchmark includes the following advantages:

**🚀 Scalability:** Easily scales to accommodate multiple large language models (LLMs), large multimodal models (LMMs), parameter-efficient fine-tuners, and diverse datasets.

**🚀 Flexibility:** Supports the flexible combination of diverse model architectures, parameter-efficient fine-tuning paradigms, and anti-forgetting methods.

**🚀 Convenience:** Enables seamless usage with a one-command "plug-and-train" interface.

**🚀 Extensibility:** Provides strong support and adaption of novel anti-forgetting methods.

**🚀 Long-Range:** Constructs the longest known multimodal continual instruction tuning benchmark: **Continual-NExT**, which contains 15 multimodal/pure-text datasets and provides comprehensive continual learning performance evaluation under **Long Term Training**.

## Contents

- [Installation](#Installation)
- [Supported Models](#Supported-Models)
- [Supported Peft Tuners](#Supported-Peft-Tuners)
- [Supported Methods](#Supported-Methods)
- [Dataset](#Dataset)
- [Training and Evaluation](#Training-and-Evaluation)
- [Evaluation Metrics](#Evaluation-Metrics)
- [Supported Dataset Formats](#Supported-Dataset-Formats)
- [Experimental Results](#Experimental-Results)
- [Acknowledgements](#Acknowledgements)
- [Future Plans](#Future-Plans)

## Installation
1. Create Conda Environment:
```shell
conda create -n continual python==3.10
conda activate continual
```

2. Install From Source:
```shell
git clone https://github.com/ECNU-SII/Continual-NExT.git
cd Continual-NExT
pip install -e .
```

3. Install Flash Attention Package:
```shell
pip install flash_attn
```
**Notice:** Considering that direct pip installation may cause exceptions, it is recommended to install flash-attn in an offline manner.

Running Environment:

|              | Range        | Recommended | Notes                                     |
| ------------ |--------------| ----------- | ----------------------------------------- |
| python       | >=3.9        | 3.10        |                                           |
| cuda         |              | cuda12      | No need to install if using CPU, NPU, MPS |
| torch        | >=2.0        |             |                                           |
| transformers | >=4.33       | 4.51.3      |                                           |
| modelscope   | >=1.23       |             |                                           |
| trl | >=0.13,<0.19 | 0.18 |RLHF|
| deepspeed    | >=0.14       | 0.14.5 / 0.16.9 | Training                                  |
| vllm         | >=0.5.1      | 0.8.5.post1       | Inference/Deployment/Evaluation           |
| lmdeploy     | >=0.5        | 0.8       | Inference/Deployment/Evaluation           |

For more optional dependencies, you can refer to [here](https://github.com/modelscope/ms-swift/blob/main/requirements/install_all.sh).

## Supported Models

| Model  |                         Model size	        |       Template          |       HF Model ID   |
| :----: | :------------------------------------------: | :---------------------: |:-----------------------: |
| [Qwen/Qwen2.5](https://modelscope.cn/models/Qwen/Qwen2.5-7B-Instruct)    | 0.5B/1.5B/3B/7B/14B/32B/72B | qwen2_5  | [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) |
| [Shanghai_AI_Laboratory/internlm2_5](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm2_5-7b-chat) | 1.8B/7B/20B | internlm2 | [Shanghai_AI_Laboratory/internlm2_5-7b-chat](https://huggingface.co/internlm/internlm2_5-7b-chat) |
| [baichuan-inc/Baichuan2](https://modelscope.cn/models/baichuan-inc/Baichuan2-7B-Chat) | 7B/13B | baichuan |[baichuan-inc/Baichuan2-7B-Chat](https://huggingface.co/baichuan-inc/Baichuan2-7B-Chat) |
| [S-Lab/LLaVA](https://modelscope.cn/models/llava-hf/llava-1.5-7b-hf) | 7B/13B | llava_v1 |[llava-hf/llava-1.5-7b-hf](https://huggingface.co/llava-hf/llava-1.5-7b-hf)|
| [DeepSeek/DeepSeek-VL](https://modelscope.cn/models/deepseek-ai/deepseek-vl-7b-chat) | 1.3B/7B | deepseek |[deepseek-ai/deepseek-vl-7b-chat](https://huggingface.co/deepseek-ai/deepseek-vl-7b-chat)|
| [Qwen/Qwen-VL](https://modelscope.cn/models/Qwen/Qwen2.5-VL-7B-Instruct) | 3B/7B | qwen2_5 |[Qwen/Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)|

For more details and models, please refer to [supported models](https://swift.readthedocs.io/en/latest/Instruction/Supported-models-and-datasets.html).

## Supported Peft Tuners

| Method |                         Description                          |                           Citation                           |
| :----: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|  Adapter   | [Adapter Tuning](./assets/tuners/Adapter.md) | [https://arxiv.org/abs/1902.00751](https://arxiv.org/abs/1902.00751) |
|  LoRA      | [LoRA Tuning](./assets/tuners/LoRA.md) | [https://arxiv.org/abs/2106.09685](https://arxiv.org/abs/2106.09685) |
|  AdaLoRA   | [AdaLoRA Tuning](./assets/tuners/AdaLoRA.md) | [https://arxiv.org/abs/2303.10512](https://arxiv.org/abs/2303.10512) |
|  MoELoRA   | [MoELoRA Tuning](./assets/tuners/MoELoRA.md) | [https://arxiv.org/abs/2403.08350](https://arxiv.org/abs/2403.08350) |
|  Prompt-Tuning | [Prompt Tuning](https://github.com/ECNU-SII/Continual-NExT/blob/main/README.md) | [https://arxiv.org/abs/2403.08691](https://arxiv.org/abs/2104.08691) |
|  Prefix-Tuning | [Prefix Tuning](https://github.com/ECNU-SII/Continual-NExT/blob/main/README.md) | [https://arxiv.org/abs/2403.00190](https://arxiv.org/abs/2101.00190) |
|  P-Tuning | [P Tuning](https://github.com/ECNU-SII/Continual-NExT/blob/main/README.md) | [https://arxiv.org/abs/2403.10385](https://arxiv.org/abs/2103.10385) |

For more details and pefts, please refer to [supported pefts](https://swift.readthedocs.io/en/latest/Instruction/Supported-models-and-datasets.html).

## Supported Methods

| Method |                         Description                          |                           Citation                           |
| :----: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|  CIA   | [Large Continual Instruction Assistant](./assets/methods/CIA.md) | [https://arxiv.org/pdf/2410.10868](https://arxiv.org/pdf/2410.10868) |
|  EWC   | [Elastic Weight Consolidation](./assets/methods/EWC.md) | [https://arxiv.org/abs/1612.00796](https://arxiv.org/abs/1612.00796) |
|  GEM   | [Gradient Episodic Memory](./assets/methods/GEM.md) | [NeurIPS 2017](https://proceedings.neurips.cc/paper/2017/file/f87522788a2be2d171666752f97ddebb-Paper.pdf) |
|  LWF   | [Learning Without Forgetting](./assets/methods/LWF.md) | [TPAMI 2017](https://ieeexplore.ieee.org/ielaam/34/8520726/8107520-aam.pdf) |
| Reply  | [Experience Replay](./assets/methods/Reply.md) | [NeurIPS 2019](https://proceedings.neurips.cc/paper_files/paper/2019/file/fa7cdfad1a5aaf8370ebeda47a1ff1c3-Paper.pdf) |

Other methods are coming soon!

## Dataset

### Continual-News

We introduce a purely Chinese text benchmark, which is constructed by collecting important current affairs news according to distinct months and years. The entire benchmark is divided into four datasets, corresponding to the years 2022, 2023, 2024, and 2025. For the datasets from 2022 to 2024, major news events from whole year are collected. The 2025 dataset, however, contains only significant news from the first half of the year. The datasets are structured as multi-turn conversational form (QA pairs). The number of QA pairs in each dataset is presented in the following table.

| Dataset  | 2022 | 2023 | 2024 | 2025 |
| -------- | ---- | ---- | ---- | ---- |
| QA Pairs | 1408 | 1246 | 1073 | 220  |

The model is continually trained in succession on the datasets of 2022, 2023, 2024, and 2025. The model is further evaluated by recomputing the accuracy on each respective trained dataset based on the model weights obtained in the final dataset of training.

### Continual-NeXT

We introduce the longest known multimodal continual instruction tuning benchmark to date, comprising a total of 15 multimodal and pure-text datasets, nearly double the number utilized in comparable studies. Specifically, the benchmark includes the following datasets: ArXivQA, GeoChat, IconQA, ClevrMath, CodeQA, ImageNet, Flickr30k, DocVQA, TextVQA, MathQA, ChartQA, PathVQA, Grounding, ScienceQA, and WikiQA. To facilitate benchmark unification, we reproduce the annotations for all datasets, standardize the training tasks into a consistent question-answering format, and design distinct instruction templates tailored to each dataset, thereby enabling efficient continual instruction tuning. Notably, the proposed benchmark is the most extensive of its kind, encompassing both pure-text and multimodal inputs. Furthermore, the included datasets span a wide range of domains and represent various distinct tasks, including image classification, code generation, remote sensing recognition, optical character recognition (OCR), visual grounding, and others. Consequently, this benchmark provides a comprehensive and rigorous platform for evaluating the effectiveness of diverse continual learning methods across different modeling paradigms. The number of QA pairs in each dataset is presented in the following table.

| Dataset  | Flickr30K | RefCOCO | ScienceQA | MathQA | CodeQA | ArxivQA | ChartQA | TextVQA |
| -------- | --------- | ------- | --------- | ------ | ------ | ------- | ------- | ------- |
| QA Pairs | 31K       | 142K    | 20K       | 37K    | 59K    | 43K     | 30K     | 40K     |

| Dataset  | ImageNet | IconQA | WikiQA | DocVQA | PathVQA | GeoVQA | Clevr Math |
| -------- | -------- | ------ | ------ | ------ | ------- | ------ | ---------- |
| QA Pairs | 135K     | 33K    | 29K    | 40K    | 33K     | 31K    | 43K        |

The model is continually trained in succession on the datasets of ArXivQA, GeoChat, IconQA, ClevrMath, CodeQA, ImageNet, Flickr30k, DocVQA, TextVQA, MathQA, ChartQA, PathVQA, Grounding, ScienceQA, and WikiQA. The model is further evaluated by recomputing the accuracy on each respective trained dataset based on the model weights obtained in the final dataset of training.

## Training and Evaluation
### Training
Notice: When opening a new terminal, please execute the following command:
```shell
export PYTHONPATH=$PYTHONPATH:peft
```

For training 'Reply' on Internlm2.5-7b
```shell
sh scripts/train/internlm_train_reply.sh
```

For training 'LWF' on Qwen2.5-7b
```shell
sh scripts/train/qwen_train_lwf.sh
```

For training 'EWC' on Internlm2.5-7b
```shell
sh scripts/train/internlm_train_ewc.sh
```

For training 'GEM' on Qwen2.5-7b
```shell
sh scripts/train/qwen_train_gem.sh
```

For training 'CIA' on Internlm2.5-7b
```shell
sh scripts/train/internlm_train_cia.sh
```

For training 'MoELoRA' on Qwen2.5-7b
```shell
sh scripts/train/qwen_train_moe.sh
```

You can change the MoELoRA expert number in ./peft/lora/moeloralayer.py Line69.

"--model" is the model path. If the file does not exist, it will be downloaded online. For specific details, please refer to [swift](https://github.com/modelscope/ms-swift).

Adapters steps calculation procedure is as follows:  
$$\text{steps} = \left\lceil \frac{\text{num-samples}}{\text{per-device-train-batch-size} \times \text{NPROC-PER-NODE}} \right\rceil \times \text{num-train-epochs}$$   
$\lceil \cdot \rceil$ denotes rounding up to the nearest integer. Subsequently, the path is "--adapters ms-swift-main/output/{ouput_dir}/2022/{steps}".

Please note that the implementation of LWF in Qwen does not support flash_attn.


### Evaluation
To calculate the performance metrics of the model results, we first need to deploy the model as a background service to ensure cantinual operation. Here's a sample on Continual-News benchmark: Use the final model trained on 2025 data to test 2022 data:  


#### 1. Deploy the model service in the background  
```shell
# Run the deployment script in the background and redirect output to a log file
nohup sh evaluation/deploy.sh &> deployment.log &
# Check the process status (replace <PID> with the actual process ID if needed)
ps -ef | grep deploy.sh
```  

- **Infer_backend configuration**:  
  The `--infer_backend` can be set to `pt` or `vllm`. For detailed instructions, refer to [swift](https://github.com/modelscope/ms-swift).  
  **Note**: MoELoRA does not support `vllm` during deployment—use `pt` instead.  


#### 2. Execute subsequent scripts in a new terminal  
After deploying the service, open a new terminal window to proceed with generating responses and calculating metrics:  

```python
# Generate model responses and save to files
python evaluation/test_ans.py
# Calculate performance metrics (e.g., similarity scores)
python evaluation/sim.py
```  


#### 3. Key considerations  
- **Background service management**:  
  - To stop the service, find the process ID with `ps -ef | grep deploy.sh` and use `kill <PID>`.  
  - Logs are stored in `deployment.log` for troubleshooting.  

- **Path modifications**:  
  Ensure to update the corresponding files and model paths in `deploy.sh`, `test_ans.py`, and `sim.py` to match your environment.  


This approach allows the service to run continually in the background while you execute evaluation scripts in a separate terminal, ensuring non-blocking workflow execution.

### Distributed Training

SWIFT originally supports distributed training by using DDP/FSDP/DeepSpeed. In our modification, we select the DeepSpeed method to implement distributed training.  The following Table shows the compatibility status of each continual learning PEFT/method with various DeepSpeed ZeRO configurations.

| Method                                 | Single GPU | ZeRO-0 | ZeRO-1 | ZeRO-2 | ZeRO-3 | ZeRO-3+Offload |
| -------------------------------------- | :--------: | :----: | :----: | :----: | :----: | :------------: |
| LoRA Fine-Tuning                       |     ✅      |   ✅    |   ✅    |   ✅    |   ✅    |       ✅        |
| MoELoRA (Mixture of Experts with LoRA) |     ✅      |   ✅    |   ✅    |   ✅    |   ✅    |       ✅        |
| Experience Replay                      |     ✅      |   ✅    |   ✅    |   ✅    |   ✅    |       ✅        |
| Learning Without Forgetting (LWF)      |     ✅      |   ✅    |   ✅    |   ✅    |   ✅    |       ✅        |
| Elastic Weight Consolidation (EWC)     |     ✅      |   ✅    |   ✅    |   🚫    |   🚫    |       🚫        |
| Continual Instruction Tuning (CIA)     |     ✅      |   ✅    |   🚫    |   🚫    |   🚫    |       🚫        |
| Gradient Episodic Memory (GEM)         |     ✅      |   ✅    |   🚫    |   🚫    |   🚫    |       🚫        |

**Legend**:

- ✅ Compatible
- 🚫 Not compatible

**Incompatibility Reasons**:

- **GEM,**  **CIA**: Requires obtaining gradient and parameters which are incompatible with ZeRO-1 and above due to the way gradients and parameters are partitioned across devices.
- **EWC**: Requires obtaining parameters which are incompatible with ZeRO-2 and above due to the way parameters are partitioned across devices.

## Evaluation Metrics

We evaluate the performance by using Accuracy (ACC) metric. Accuracy are calculated according to specific downstrem tasks.

### For Single-Choice Question
Accuracy is obtained by judging whether Answer of LMMs equals to Ground Truth.

### For Fill-Blank Question
Accuracy is obtained by judging whether Answer of LMMs equals to Ground Truth or whether Ground Truth is concluded in Answer of LMMs.

### For Long-Answer Question 
Accuracy is obtained with the following steps:

1. **Encoding with BERT**
    We use a pretrained multilingual BERT model (e.g., `paraphrase-multilingual-MiniLM-L12-v2`) to convert each assistant reply and corresponding ground truth into a high-dimensional vector (embedding) that captures the **semantic meaning** of the text.

2. **Cosine Similarity Calculation**
    For each pair of replies and ground truths, we compute the **cosine similarity** between their embeddings. This value ranges from 0 (irrelated meaning) to 1 (identical meaning), with values close to 1 indicating that the two responses are semantically very similar.

3. **Output**
    We print the similarity score for each matched pair and compute the **average similarity score across all pairs**, which gives a quantitative measure of how semantically similar the assistant responses are between the replies and ground truths.

In conclusion, Accuracy can be calculated as:

$$\text{Similarity}(A,B)=\text{cos}(v_A,v_B)=\frac{v_A·v_B}{∥v_A∥·∥v_B∥}$$

$v_A$: The BERT embedding vector of text A

$v_B$: The BERT embedding vector of text B

$·$: Dot product of the two vectors

$∥⋅∥$: L2 norm (i.e., length) of the vector

$cos⁡(v_A,v_B)$: Cosine similarity between vectors A and B, ranging from 0 to 1

**Average Accuracy (Avg.ACC)** is used for averaging the test accuracy of all datasets, which represents the comprehensive performance of continual tuning.

$$\text{Average Accuracy} = \frac{1}{T}\sum_{i=1}^{T}A_{T,i},$$

**Forgetting (FOR)** is utilized to indicate the test accuracy reduction of past datasets after learning the new dataset, which denotes the stability performance.

$$\text{Forgetting} = \frac{1}{T-1}\sum_{i=1}^{T-1}{A_{T,i} – \text{max}(A_{j,i})_{j \in [i,T-1]}},$$

**New Accuracy (New.ACC)** is employed to average the test accuracy of new datasets, which refers to the plasticity performance.

$$\text{New Accuracy} = \frac{1}{T}\sum_{i=1}^{T}A_{i,i},$$

where $T$ is the number of datasets, $A_{T,i}$ is the accuracy of $i$-th dataset on the model trained after $T$-th dataset, $A_{j,i}$ is the accuracy of $i$-th dataset on the model trained after $j$-th dataset, and $A_{i,i}$ is the accuracy of $i$-th dataset on the model trained after $i$-th dataset.

## Supported dataset formats
Messages format (standard format):
```python
{"messages": [{"role": "system", "content": "<system>"}, {"role": "user", "content": "<query1>"}, {"role": "assistant", "content": "<response1>"}, {"role": "user", "content": "<query2>"}, {"role": "assistant", "content": "<response2>"}]}
```
ShareGPT format:
```python
{"system": "<system>", "conversation": [{"human": "<query1>", "assistant": "<response1>"}, {"human": "<query2>", "assistant": "<response2>"}]}
```

Alpaca format:
```python
{"system": "<system>", "instruction": "<query-inst>", "input": "<query-input>", "output": "<response>"}
```

Query-Response format:
```python
{"system": "<system>", "query": "<query2>", "response": "<response2>", "history": [["<query1>", "<response1>"]]}
```

For more details, please refer to [swift datasets](https://swift.readthedocs.io/en/latest/Customization/Custom-dataset.html).

## Experimental Results

We implemented two parameter efficient fine-tunings (*i.e.* LoRA and MoELoRA), and five continual learning methods (namely Replay, LWF, EWC, GEM and CIA based on LoRA fine-tuning) on our proposed **Continual-News** dataset. Results are shown in the following two Tables.

**Continual-News Results on InternLM2.5-7b-chat**

| Method  | 2022  | 2023  | 2024  |  2025  | Avg.ACC | Forgetting | New.ACC |
| :-----: | :---: | :---: | :---: | :----: | :-----: | :--------: | :-----: |
|  LoRA   | 66.58 | 61.88 | 75.45 |  100   |  75.98  |   29.63    |  98.2   |
| MoELoRA | 65.34 | 59.26 | 71.63 |  100   |  74.06  |   34.59    |   100   |
| RePlay  | 78.04 | 69.19 | 83.71 | 99.84  |  82.70  |   22.18    |  99.33  |
|   LWF   | 72.25 | 63.53 | 80.18 | 100.00 |  78.99  |   27.59    |  99.58  |
|   EWC   | 68.55 | 61.23 | 76.97 |  100   |  76.69  |   29.26    |  98.63  |
|   GEM   | 76.19 | 71.52 | 87.83 |  100   |  83.89  |   19.32    |  98.38  |
|   CIA*  | 74.25 | 65.71 | 82.96 |  100   |  80.73  |   23.64    |  98.46   |


**Continual-News Results on Qwen2.5-7b**

| Method  | 2022  | 2023  | 2024  |  2025  | Avg.ACC | Forgetting | New.ACC |
| :-----: | :---: | :---: | :---: | :----: | :-----: | :--------: | :-----: |
|  LoRA   | 67.13 | 63.16 | 76.51 |  100   |  76.7   |   31.02    |  99.97  |
| MoELoRA | 65.26 | 61.52 | 72.32 |  100   |  74.78  |   33.62    |  99.99  |
| RePlay  | 77.90 | 68.43 | 81.14 | 100.00 |  81.87  |   24.18    |   100   |
|   LWF   | 73.79 | 65.31 | 81.93 | 99.94  |  80.24  |   24.51    |  98.62  |
|   EWC   | 69.73 | 61.98 | 77.37 |  100   |  77.27  |   29.95    |  99.74  |
|   GEM   | 75.82 | 72.03 | 88.67 |  100   |  84.13  |   21.04    |  99.91  |
|   CIA*  | 74.06 | 67.15 | 82.58 |  100   |  80.95  |   25.40    |   100   |

CIA* denotes we adopt the CIA method without instruction grouping mechanism.

Additionally, we also present a case (shown in the following Figure) that illustrates the continual knowledge update of LLMs.

![demo](assets/demo.png)

In addition, we also implemented two parameter efficient fine-tunings (*i.e.* LoRA and MoELoRA), and six continual learning methods (namely Replay, LWF, EWC, GEM and CIA based on LoRA fine-tuning) on our proposed **Continual-NExT** benchmark (including 15 multimodal/pure text datasets, forming a **Long Term** order). Results are shown in the following Table.

**Continual-News Results on LLaVA-7b**

|  Method  | ArxivQA | GeoChat | IconQA | ClevrMath | CodeQA | ImageNet | Flickr30k |
| :------: | :-----: | :-----: | :----: | :-------: | :----: | :------: | :-------: |
| Pretrain |  36.99  |  67.67  | 18.77  |   20.27   |  0.26  |   18.1   |   17.27   |
|   LoRA   |  53.99  |  92.23  | 47.23  |   44.86   |  4.36  |  67.84   |   17.16   |
|   EWC    |  55.16  |  91.73  | 47.17  |   49.3    |  4.38  |  82.03   |   16.71   |
|   GEM    |  55.3   |  91.03  | 49.13  |   48.3    |  4.76  |   76.2   |   16.21   |
|   LWF    |  51.04  |  87.33  | 30.97  |   39.2    |  4.74  |  84.89   |   16.26   |
|  Replay  |  54.85  |  94.4   | 51.73  |   40.07   |  4.48  |  94.61   |   9.36    |
| MoELoRA  |   56    |  91.36  | 48.76  |   48.9    |  3.82  |  82.19   |   17.77   |

| DocVQA | TextVQA | MathQA | ChartQA | PathVQA | Grounding | ScienceQA | WikiQA |
| :----: | :-----: | :----: | :-----: | :-----: | :-------: | :-------: | :----: |
| 14.58  |  57.39  |  0.44  |   9.6   |  33.29  |   28.28   |   66.19   | 17.54  |
| 16.47  |  47.7   |  33.8  |  18.04  |  50.98  |   69.52   |   89.46   | 22.27  |
| 16.88  |  51.73  | 35.41  |   19    |  50.92  |   69.92   |   89.51   | 24.17  |
| 15.85  |  51.33  | 35.28  |  17.68  |  51.38  |   67.23   |   89.86   | 23.85  |
| 16.56  |  54.09  | 30.05  |  18.64  |  52.79  |   64.11   |   87.95   | 24.96  |
| 14.65  |  54.7   | 31.42  |  14.4   |  49.64  |   56.98   |   85.62   | 23.85  |
| 16.33  |  59.51  | 34.17  |  18.52  |  49.04  |   67.65   |   88.28   | 22.59  |

| Avg.ACC | Forgetting | New.ACC |
| :-----: | :--------: | :-----: |
|  27.11  |     -      |    -    |
|  45.06  |   11.62    |  55.91  |
|  46.93  |    9.72    |  56.01  |
|  46.23  |   10.19    |  55.74  |
|  44.24  |   12.29    |  55.70  |
|  45.38  |   11.41    |  56.03  |
|  46.99  |    8.06    |  54.51  |

## Acknowledgements

Continual-NExT is built upon the [SWIFT](https://github.com/modelscope/ms-swift), an excellent open-source framework developed by the ModelScope team. We extend our sincere gratitude for their outstanding contributions. SWIFT’s flexible and modular architecture has been instrumental in enabling the development of continual learning systems: Continual-NExT.

Before using Continual-NExT, we highly recommend familiarizing yourself with SWIFT by consulting its [README (English version)](https://github.com/modelscope/ms-swift/blob/main/README.md), [README-CN (Chinese version)](https://github.com/modelscope/ms-swift/blob/main/README_CN.md), and its comprehensive [documentation](https://swift.readthedocs.io/en/latest/index.html). These resources provide valuable insights into SWIFT’s core design principles and implementation details, which will greatly facilitate a deeper understanding and more effective usage of Continual-NExT.

## Future Plans

• ~~We will publish a complex and hard continual tuning/evolution benchmark for **multimodal understanding** **MLLMs** with various architecture, PEFT and continual learning method.~~

• We will publish a novel and challenge continual tuning/evolution benchmark for **Any-to-Any MLLMs** with various architecture, PEFT and continual learning method.
