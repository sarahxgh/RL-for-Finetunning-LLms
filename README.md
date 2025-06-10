# 🧪 LLM-Toxicity-Mitigation

This repository contains the full implementation for our project on mitigating toxicity in large language models (LLMs) using three alignment methods: **Supervised Fine-Tuning (SFT)**, **Proximal Policy Optimization (PPO)**, and **Direct Preference Optimization (DPO)**. The project uses the [TinyLlama-1.1B](https://huggingface.co/csarron/TinyLlama-1.1B-Chat-v1.0) model and evaluates its performance on toxic prompts after training with each method.

We rely on the [`unalignment/toxic-dpo-v0.2`](https://huggingface.co/datasets/unalignment/toxic-dpo-v0.2) dataset and [Detoxify](https://github.com/unitaryai/detoxify) as a reward and evaluation signal.

---

## 📂 Repository Structure

| File/Folder         | Description                                                                 |
|---------------------|-----------------------------------------------------------------------------|
| `DataPrep.ipynb`    | Prepares and formats the dataset for all methods, including prompt extraction and tokenization. |
| `SFT.ipynb`         | Implements Supervised Fine-Tuning using LoRA adapters and chosen responses. |
| `PPO.ipynb`         | Trains the model using PPO with rewards derived from Detoxify (reward = 1 - toxicity). |
| `DPO.ipynb`         | Applies Direct Preference Optimization using chosen vs. rejected samples.   |
| `Report/`           | Contains the LaTeX source files, charts, and bibliography for the final report. |

---

## 📊 Summary of Results

| Model              | Avg. Toxicity ↓ |
|--------------------|-----------------|
| Original (no tuning) | 0.4095         |
| SFT                | **0.0806**       |
| PPO                | 0.2295           |
| DPO                | 0.3672           |

> SFT achieved the lowest toxicity, followed by PPO. DPO offered moderate improvements.

---

## 🛠 Installation

Make sure to use Python 3.10+ and install dependencies via:

```bash
pip install -r requirements.txt
