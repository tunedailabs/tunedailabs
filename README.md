# TunedAI Labs — Causal Reasoning Benchmark

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tunedailabs/tunedailabs/blob/main/causal_reasoning_demo.ipynb)

We fine-tuned Qwen 2.5-7B on causal reasoning and scored **96.96%** on the CLadder benchmark. GPT-4o scores ~72% on the same test. Base Qwen scores ~62%.

**This repo lets you verify that claim yourself — for free, no setup required.**

---

## Run It Yourself — Step by Step

You do not need to install anything. You just need a Google account.

**Step 1** — Click the blue **"Open in Colab"** badge above.

**Step 2** — In the menu at the top of the page, click **Runtime → Change runtime type**. Set **Hardware accelerator** to **T4 GPU**. Click Save.

**Step 3** — Click **Runtime → Run all** (or press Ctrl+F9 on Windows / Cmd+F9 on Mac).

**Step 4** — Wait about 3 minutes while the models load. Progress will appear below each cell.

**Step 5** — Watch the benchmark run in real time. Each question shows both models answering side by side with a running score.

**That's it.** No coding required. You are running real AI models on Google's free cloud computers.

---

## What You Are Testing

The notebook runs a stratified sample from the real **CLadder benchmark** — the same 10,112-question dataset used to establish the 96.96% score.

Questions use fictional variable names (yupt, jyka, kwox, glimx, etc.) — the models cannot recall answers from pretraining. Answering correctly requires actual causal reasoning across three levels of Pearl's causal hierarchy:

| Rung | Type | Example |
|---|---|---|
| 1 | Association | Is X correlated with Y? |
| 2 | Intervention | If we do X, does Y change? |
| 3 | Counterfactual | If X had been different, would Y have changed? |

The default run is 20 questions (~3 minutes). Change `N_QUESTIONS = 20` to `N_QUESTIONS = 200` in the notebook for a more thorough test (~20 minutes).

---

## The Benchmark Results

| Model | CLadder Score | Notes |
|---|---|---|
| **TunedAI Labs Causal Model** | **96.96%** | Fine-tuned on causal reasoning |
| GPT-4o | ~72% | General purpose |
| Claude 3.5 Sonnet | ~68% | General purpose |
| Base Qwen 2.5-7B | ~62% | Same model, no fine-tuning |

The benchmark is public: [CLadder on GitHub](https://github.com/causalNLP/cladder). You can verify our score independently.

---

## Share Your Results

After running the notebook, open a [GitHub Issue](https://github.com/tunedailabs/tunedailabs/issues/new) and paste what you saw.

---

## About TunedAI Labs

We fine-tune open-source LLMs for real-world reasoning tasks.

**Want this for your domain?** → [tunedailabs.com](https://tunedailabs.com)
