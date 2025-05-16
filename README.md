# AMQA: Adversarial Medical Question Answering Dataset

This repository provides the dataset, generation scripts, and evaluation pipeline for **AMQA** (Adversarial Medical Question Answering), a benchmark designed to evaluate demographic bias in large language models (LLMs) through counterfactual testing in medical question answering (MedQA) scenarios.

---

## 📘 Overview

AMQA contains 801 clinical vignettes adapted from the USMLE-style medical QA setting. Each vignette includes:

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
🔹️️generate_variants/          # Multi-agent pipeline for adversarial variant generation
🔹️️benchmark_models/           # Querying LLMs and storing model answers
🔹️️analyze_results/            # Accuracy, fairness, and significance evaluation

🔹️ Results/                        # Dataset and LLM evaluation results
🔹️️AMQA_dataset.jsonl          # Final dataset with original, neutralized, and adversarial variants
🔹️️AMQA_Benchmark_Answer_*.jsonl  # Raw model predictions
🔹️️AMQA_Benchmark_Summary_*.jsonl # Accuracy and bias statistics

🔹️ figures/                        # Diagrams and visualizations (e.g., generation pipeline)

🔹️ requirements.txt                # Python dependencies
🔹️ README.md
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/XY-Showing/AMQA.git
cd AMQA
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Generate adversarial variants (optional)

```bash
python Scripts/generate_variants/run_generation_pipeline.py
```

### 4. Evaluate LLMs on the dataset

```bash
python Scripts/benchmark_models/run_evaluation.py --model gpt-4
```

### 5. Analyze results and compute bias

```bash
python Scripts/analyze_results/analyze_accuracy_gap.py
python Scripts/analyze_results/run_mcnemar_test.py
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


