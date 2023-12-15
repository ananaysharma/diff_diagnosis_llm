# Fine-Tuning Language Models for Differential Diagnosis in Healthcare

## Overview

This project focuses on refining the process of differential diagnosis within the medical domain by fine-tuning Language Models (LLMs). Through the utilization of the Low-Rank Adaptation of Large Language Models (LoRA) training approach, we aim to optimize computational resources while enhancing the accuracy of differential diagnoses. Our objective is to mitigate time and resource constraints inherent in fine-tuning private LLMs for healthcare applications.

## Related Work

### [MediTRON & ChatDoctor](link)

MediTRON's training setup for healthcareMagic-100k utilized 6 A100 GPUs over 3 hours, employing external knowledge bases for inference retrieval. ChatDoctor constructed databases using MedlinePlus and Wikipedia, extending their framework to reliable online databases for contemporary medical query evaluation.

### [BioGPT](link)

BioGPT leveraged the GPT-2 architecture across diverse medical NLP tasks, showcasing its versatility beyond question answering (QA) tasks.

## Datasets Explored

Our research began by leveraging datasets like the MIMIC III dataset and expanded into utilizing the MedQA United States Medical License Exam training data and the Clinical Practice Guideline corpus released by the Meditron team. These datasets offered comprehensive textual granularity, crucial for completion training and conversation fine-tuning our model on MedQA and PubMedQA data.

[Base Official Dataset](https://huggingface.co/datasets/pubmed_qa)
  - 211k rows
  - Columns: `pubid`, `question`, `context` (sequence), `long_answer`, `final_decision`

### PubMedQA Evaluation

Evaluating on the PubMedQA dataset required formatting model results into a specific JSON format - key is PMID and value is one of "yes", "no", and "maybe". We employed an evaluation script to measure model performance against ground truth data.




