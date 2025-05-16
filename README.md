# AMQA: Adversarial Medical Question Answering Dataset

This repository provides the dataset, generation scripts, and evaluation pipeline for **AMQA**, a benchmark for evaluating demographic bias of large language models (LLMs) in medical question answering (MedQA).

## Overview

AMQA is created from USMLE-style multiple-choice questions. Each sample includes:

* An original clinical vignette
* A neutralized clinical vignette with sensitive attributes removed
* Six adversarial variants targeting:

  * Race (Black vs. White)
  * Gender (Male vs. Female)
  * Socioeconomic Status (Low vs. High Income)

Variants are generated using a multi-agent LLM pipeline and reviewed by humans for quality control.

## Repository Structure

```
AMQA/
├── Scripts/                         # Generation, evaluation, and analysis scripts
│   ├── AMQA_generation_batch/
│   ├── AMQA_Benchmark_LLM/
│   └── .../
├── Results/                         # Dataset and benchmark results
│   ├── AMQA_dataset.jsonl
│   ├── AMQA_Benchmark_Answer_*.jsonl
│   └── AMQA_Benchmark_Summary_*.jsonl
└── README.md
```



## Evaluation Metrics

* **Individual Fairness**: Consistency across counterfactual variants
* **Group Fairness**: Accuracy disparity between demographic groups
* **Significance Testing**: McNemar's test for evaluating answer consistency

## Dataset

* `AMQA_dataset.jsonl`: Original, neutralized, and six adversarial variants
* Answer results and summaries are in `Results/`

