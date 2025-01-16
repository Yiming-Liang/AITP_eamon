# Aligning Instruction Tuning with Pre-training
This repository contains codes and scripts for the [AITP paper](https://arxiv.org/abs/2408.08072), which enriches the diversity and optimizes the distribution of existing datasets by using the pre-training corpus as a reference distribution.

## Introduction
Instruction tuning enhances large language models (LLMs) to follow human instructions across diverse tasks, relying on high-quality datasets to guide behavior. However, these datasets, whether manually curated or synthetically generated, are often narrowly focused and misaligned with the broad distributions captured during pre-training, limiting LLM generalization and effective use of pre-trained knowledge. We propose *Aligning Instruction Tuning with Pre-training* (AITP), a method that bridges this gap by identifying coverage shortfalls in instruction-tuning datasets and rewriting underrepresented pre-training data into high-quality instruction-response pairs. This approach enriches dataset diversity while preserving task-specific objectives.

![The Pipeline of AITP. ](./pictures/AITPpipeline.png)
<center>The Pipeline of AITP.</center>

## Background



## Usage
This is an unoptimized version of the code, which is the same as what we used during the actual run.

1. Generate the difference set
```
cd ./AITP
sh scripts/difference_general_GPU_bge_batch_state.sh
```
2. Rewrite raw text into instruction-response pairs


3. Training

## citation
```
@inproceedings{Liang2024isheep,
  title  = {I-SHEEP: Self-Alignment of LLM from Scratch through an Iterative Self-Enhancement Paradigm},
  author = {Yiming Liang and Ge Zhang and Xingwei Qu and Tianyu Zheng and Jiawei Guo and Xeron Du and Zhenzhu Yang and Jiaheng Liu and Chenghua Lin and Lei Ma and Wenhao Huang and Jiajun Zhang},
  year   = {2024},
  url    = {https://openreview.net/forum?id=y8Ng9dkuK9&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DAAAI.org%2F2025%2FAI_Alignment_Track%2FAuthors%23your-submissions)}
}
```
