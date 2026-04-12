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

The notebook generates **fresh questions the model has never seen** — correct answers are computed from the numbers, not recalled from training. This rules out memorization entirely.

Questions use fictional variable names (yupt, jyka, kwox, glimx, etc.) across three levels of difficulty:

| Level | Type | Example |
|---|---|---|
| 1 | Association | Is X correlated with Y? |
| 2 | Intervention | If we do X, does Y change? |
| 3 | Deep Causal | If X had been different, would Y have changed? |

The default run is 200 questions (~20 minutes on T4).

---

## Verified Results — 200 Fresh Questions

These numbers were produced on questions **generated at runtime** — the model was not trained on them. Correct answers are computed from the given probabilities.

| Model | Overall | Level 1 | Level 2 | Level 3 |
|---|---|---|---|---|
| **TunedAI Labs ★** | **93.0%** | **85%** | **100%** | **100%** |
| Base Qwen 2.5-7B | 64.0% | 61% | 78% | 44% |
| **Gap** | **+29 pp** | +24 pp | +22 pp | +56 pp |

Level 3 (the hardest question class) is where the gap is largest: the tuned model answers every question correctly while the base model performs near chance.

The CLadder benchmark score (96.96% on 10,112 questions) is also public: [CLadder on GitHub](https://github.com/causalNLP/cladder).

---

## Common Questions

**"Isn't this just overfitting to CLadder?"**

The notebook generates questions at runtime using fictional variable names the model has never seen — yupt, jyka, kwox, glimx. Correct answers are computed from the probability parameters in the question, not retrieved from any corpus. The tuned model scores 93% overall and 100% on Level 3. The base model scores 64% on the same questions. You can't memorize questions that didn't exist until the moment you ran the notebook.

**"Isn't this benchmaxxing?"**

The model was not trained on CLadder questions. Training data was synthetically generated with machine-verified answers derived from explicit probability parameters — not sourced from CLadder or any public causal dataset. There's also a keyword-scrubbed version of CLadder that removes the 168 questions where the answer can be guessed from phrases like "collider bias" — no causal reasoning required. The score holds on that version too.

**Short version:** The notebook generates fresh questions on the spot with fictional variables. Run it. Check the numbers yourself.

---

## Share Your Results

After running the notebook, open a [GitHub Issue](https://github.com/tunedailabs/tunedailabs/issues/new) and paste what you saw.

---

## About TunedAI Labs

We fine-tune open-source LLMs for real-world reasoning tasks.

**Want this for your domain?** → [tunedailabs.com](https://tunedailabs.com)
