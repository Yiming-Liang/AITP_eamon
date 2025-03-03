# Aligning Instruction Tuning with Pre-training
This repository contains codes and scripts for the [AITP paper](https://arxiv.org/abs/2408.08072), which enriches the diversity and optimizes the distribution of existing datasets by using the pre-training corpus as a reference distribution.

## Introduction
Instruction tuning enhances large language models (LLMs) to follow human instructions across diverse tasks, relying on high-quality datasets to guide behavior. However, these datasets, whether manually curated or synthetically generated, are often narrowly focused and misaligned with the broad distributions captured during pre-training, limiting LLM generalization and effective use of pre-trained knowledge. We propose *Aligning Instruction Tuning with Pre-training* (AITP), a method that bridges this gap by identifying coverage shortfalls in instruction-tuning datasets and rewriting underrepresented pre-training data into high-quality instruction-response pairs. This approach enriches dataset diversity while preserving task-specific objectives.

![The Pipeline of AITP. ](./pictures/AITPpipeline.png)
<center>The Pipeline of AITP.</center>


## Usage
This is an unoptimized version of the code, which is the same as what we used during the actual run.

1. **Generate the difference set**
```
cd ./AITP
sh scripts/difference_general_GPU_bge_batch_state.sh
```

2. **Rewrite raw text into instruction-response pairs**

   We leverage the [KOR-Bench](https://github.com/KOR-Bench/KOR-Bench) codebase to rewrite raw text into instruction-response pairs. To do so, place the YAML files from the `AITP/prompts` directory into the `KOR-Bench/config/prompt` directory. You will need to follow the KOR-Bench usage instructions for data generation. Specifically, we provide three YAML files that you will use in sequence for distillation:
   
   - `generate_shift_answer.yaml`
   - `score_shift.yaml`
   - `rewrite_shift_question.yaml`

   These YAML files will guide the process of converting raw text into high-quality instruction-response pairs, enriching the dataset to better align instruction tuning with pre-training data.


3. **Training**

   For training the model, we utilize the open-source [Llama Factory](https://github.com/hiyouga/LLaMA-Factory) framework. This framework provides the necessary tools for fine-tuning the model. The specific training parameters for each model are detailed in the paper. To use the framework, follow the setup and execution instructions provided in the Llama Factory repository. 

   The training process will leverage the instruction-response pairs generated in the previous step, and fine-tune the model according to the predefined training parameters to enhance performance in alignment with the pre-training data.

## citation
```
@article{liang2025aligning,
  title   = {Aligning Instruction Tuning with Pre-training},
  author  = {Yiming Liang and Tianyu Zheng and Xinrun Du and Ge Zhang and Xingwei Qu and Xiang Yue and Chujie Zheng and Jiaheng Liu and Lei Ma and Wenhu Chen and Guoyin Wang and Zhaoxiang Zhang and Wenhao Huang and Jiajun Zhang},
  year    = {2025},
  journal = {arXiv preprint arXiv: 2501.09368}
}
```
