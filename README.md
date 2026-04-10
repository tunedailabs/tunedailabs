# TunedAI — Causal Reasoning Benchmark

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mgentry11/tunedai/blob/main/causal_reasoning_demo.ipynb)

We fine-tuned Qwen 2.5-7B on causal reasoning and scored **96.96%** on the CLadder benchmark. GPT-4o scores ~72% on the same test. Base Qwen scores ~62%.

**This repo lets you verify that claim yourself — for free, no setup required.**

---

## Run It Yourself — Step by Step

You do not need to install anything. You just need a Google account.

**Step 1** — Click the blue **"Open in Colab"** badge above.

**Step 2** — In the menu at the top of the page, click **Runtime → Change runtime type**. Set **Hardware accelerator** to **T4 GPU**. Click Save.

**Step 3** — Click **Runtime → Run all** (or press Ctrl+F9 on Windows / Cmd+F9 on Mac).

**Step 4** — Wait about 2 minutes while the models load. You will see progress messages appear below each cell.

**Step 5** — Scroll down to see the results. Each test question is answered by four models side by side.

**That's it.** No coding required. You are running real AI models on Google's free cloud computers.

---

### Want to compare against GPT-4 and Claude too?

The notebook already includes GPT-4o and Claude 3.5 Sonnet columns. To activate them you need API keys:

- **OpenAI key** (for GPT-4o): sign up at platform.openai.com → API keys → Create new key
- **Anthropic key** (for Claude): sign up at console.anthropic.com → API keys → Create key

Paste your keys into the cell near the top of the notebook that looks like this:

```
OPENAI_API_KEY    = ""   <- paste your key between the quotes
ANTHROPIC_API_KEY = ""   <- paste your key between the quotes
```

If you leave them blank those columns will say "No API key provided" and the rest still runs fine.

---

## What You Are Testing

The notebook asks six questions drawn from classic texts — all written before AI existed. The correct answers were established by human experts decades ago. This rules out the models just pattern-matching from the internet.

| Question | Original Source | Year |
|---|---|---|
| Kidney stone treatment paradox (Simpson's Paradox) | E.H. Simpson | 1951 |
| Cholera and the Broad Street water pump | John Snow | 1854 |
| Handwashing and childbed fever | Ignaz Semmelweis | 1847 |
| Smoking and lung cancer — no experiment was run | Bradford Hill | 1965 |
| Billiard balls and the limits of cause-and-effect logic | David Hume | 1748 |
| Why random assignment changes what we can conclude | R.A. Fisher | 1935 |

There is also a cell at the bottom where you can type your own question.

---

## The Benchmark Results

| Model | Score | Notes |
|---|---|---|
| **TunedAI Causal Model** | **96.96%** | Fine-tuned on causal reasoning |
| GPT-4o | ~72% | General purpose |
| Claude 3.5 Sonnet | ~68% | General purpose |
| Base Qwen 2.5-7B | ~62% | Same model, no fine-tuning |

The benchmark is public: [CLadder on GitHub](https://github.com/causalNLP/cladder). You can verify our score independently.

---

## Share Your Results

After running the notebook, open a [GitHub Issue](https://github.com/mgentry11/tunedai/issues/new) and paste what you saw. Tell us:

- Which questions the TunedAI model got right that the others got wrong
- Anything surprising
- Your own question and what happened

We read every submission and will post a summary of independent results.

---

## About TunedAI

We fine-tune open-source LLMs for real-world reasoning tasks.

**Want this for your domain?** → [tunedai.ai](https://tunedai.ai)
