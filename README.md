# AMQA: Adversarial Medical Question Answering Dataset

This repository provides the dataset, generation scripts, and evaluation pipeline for **AMQA** (Adversarial Medical Question Answering), a benchmark designed to evaluate demographic bias in large language models (LLMs) through counterfactual testing in medical question answering (MedQA) scenarios.

---

## 📘 Overview

AMQA contains 4,806 clinical vignettes adapted from the USMLE-style medical QA setting. Each vignette includes:

* A **neutralized** version with sensitive attributes removed
* **Six adversarial variants** targeting:

  * Race (Black vs. White)
  * Gender (Male vs. Female)
  * Socioeconomic Status (Low vs. High Income)

These variants are generated using a multi-agent adversarial prompting framework and evaluated for bias-triggering behavior across state-of-the-art LLMs. The pipeline combines LLM automation with human review to ensure quality and reliability.

---

## 📁 Repository Structure

```text
AMQA/
🔹️ Scripts/                         # All scripts for generation, evaluation, and analysis
🔹️️AMQA_generation_batch/          # Multi-agent pipeline for adversarial variant generation
🔹️️AMQA_Benchmark_LLM/           # Querying LLMs and storing model answers and statistical analyze the result
🔹️️analyze_results/            # Accuracy, fairness, and significance evaluation

🔹️ Results/                        # Dataset and LLM evaluation results
🔹️️AMQA_dataset.jsonl          # Final dataset with original, neutralized, and adversarial variants
🔹️️AMQA_Benchmark_Answer_*.jsonl  # Raw model predictions
🔹️️AMQA_Benchmark_Summary_*.jsonl # Accuracy and bias statistics


🔹️ README.md
```

---
## 📊 Evaluation Metrics

We adopt **individual fairness** and **group fairness**, two widely studied fairness notions in AI\~\cite{chen2024fairness}:

* **Individual Fairness**: Measures counterfactual consistency — models should produce similar outputs for similar patients differing only in sensitive attributes.
* **Group Fairness**: Measures statistical disparities in accuracy across demographic groups.
* **Statistical Significance**: McNemar's test is used to evaluate whether answer differences between counterfactual pairs are significant.

---

## 📦 Dataset Contents

The full AMQA dataset is provided in:

* `Results/AMQA_dataset.jsonl`

Each entry contains:

* Original vignette
* Neutralized version
* 6 adversarial variants (based on Race, Gender, SES)
* Multiple-choice options and ground truth

Model outputs and statistical summaries can be found in:

* `Results/AMQA_Benchmark_Answer_*.jsonl`
* `Results/AMQA_Benchmark_Summary_*.jsonl`

---

## 🧪 Scripts

The `Scripts/` directory contains all code for dataset construction, model evaluation, and result analysis:

* `generate_variants/`: Multi-agent pipeline (Generation-Agent, Fusion-Agent, Evaluation-Agent) to generate adversarial examples.
* `benchmark_models/`: Code to query LLM APIs (e.g., GPT-4, Claude, Gemini, Deepseek, Qwen) and store predictions.
* `analyze_results/`: Computes accuracy, bias metrics, and performs statistical testing (e.g., McNemar’s test).

---

## 📌 Citation


