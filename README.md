<p align="center">
  <img src="assets/clustering_diagram.svg" alt="Clustering model evaluation and selection pipeline" width="100%">
</p>

<p align="right"><a href="README.ru.md">Русская версия →</a></p>

# Multi-Level Clustering for Chatbot Intent Classification

![Python](https://img.shields.io/badge/Python-scikit--learn-14131a?style=flat-square&labelColor=14131a&color=7a1f2b)
![SQL](https://img.shields.io/badge/SQL-data%20pipeline-14131a?style=flat-square&labelColor=14131a&color=7a1f2b)
![LLM](https://img.shields.io/badge/LLM-annotation-14131a?style=flat-square&labelColor=14131a&color=7a1f2b)
![status](https://img.shields.io/badge/status-production-14131a?style=flat-square&labelColor=14131a&color=c9a227)

> **Note on source code and scope.** This repository documents methodology, design decisions, and results only; the implementation is proprietary to the employer and not published here. This was a cross-functional initiative under a single product owner, run jointly with the NLP and chatbot engineering teams. What's described here is the data analytics side of the work — evaluation methodology, experimental design, and model selection — which the author led. The fine-tuned LLM checkpoint used as one of the four candidate models was trained by the ML engineering team, not by the author.

A flexible, multi-level clustering system for chatbot queries that replaced a single rigid clustering approach with four candidate methods evaluated in parallel — by blind human judgment and by automated clustering metrics — so the best-performing model could be chosen on evidence rather than intuition.

## At a glance

| | |
|---|---|
| **Candidate approaches compared** | 4 (classical TF-IDF, hybrid, LLM-enhanced, fine-tuned LLM checkpoint) |
| **Evaluation methods** | blind human rating + automated `sklearn` metrics (Purity, Inverse Purity, F-score, AMI), run independently |
| **Selection outcome** | one model ranked highest across all four automated metrics |
| **Scale** | applied to the full incoming stream of chatbot queries requiring clustering, not a fixed one-off sample |
| **Status** | integrated into the company's internal chatbot platform; in production since early 2025 |

## Problem

The existing clustering system had three compounding issues: clusters frequently mixed unrelated intents together, readability collapsed as data volume grew, and complex conditional intents — the ones that depend on more than one signal to classify correctly — were essentially assigned at random.

## Approach

**Metric design (with the NLP team).** Before comparing any models, the team defined what "good clustering" actually meant, in three measurable terms: homogeneity (a cluster should represent one intent, not several), separability (clusters should be clearly distinguishable from each other), and practical applicability (a cluster shouldn't need extensive manual cleanup before it's usable). This gave the comparison an objective basis instead of a subjective "this one looks better" judgment.

**Four candidate approaches**, built to handle raw, partially processed, and incomplete queries alike:
- **Classical** — TF-IDF combined with semantic similarity.
- **Hybrid** — tags, existing annotations, and frequency-based features.
- **LLM-enhanced** — an LLM layer added on top of clustering output for extra interpretation.
- **Fine-tuned LLM checkpoint** — trained by the ML engineering team specifically for this task.

**Evaluation, run two ways in parallel.** Human raters scored clustering quality on the same dataset without knowing which model produced which output (blind evaluation), while `sklearn`-based metrics (Purity, Inverse Purity, F-score, AMI) scored the same outputs automatically. The two methods were designed to check each other — a model that only looked good by one measure wouldn't have been trusted.

**Re-annotation with LLMs.** Once a model was selected, an LLM-based re-annotation pass was added to sharpen cluster interpretability specifically for the more complex, conditional intents that the old system struggled with.

## Results

- The selected model outperformed the three alternatives across all four automated evaluation metrics, with blind human ratings pointing to the same conclusion.
- Classification accuracy improved specifically on complex, multi-condition intents — the exact category the old system handled worst.
- The clustering approach is flexible enough to run on raw, partially processed, or incomplete queries without a separate pipeline for each case.
- Reduced the amount of manual cleanup previously needed to make clusters usable.

## Business impact

- Better clustering fed directly into more relevant chatbot responses for end users.
- Reduced manual review workload for the intent analytics team.
- Established a reusable evaluation framework — blind rating plus automated clustering metrics — that the team could apply to future model comparisons, not just this one.

## Tech stack

Python · scikit-learn · SQL · LLM-based annotation · A/B and blind-testing frameworks · BI tools

---

<sub>Cross-functional project (analytics, NLP, engineering) under shared product ownership. Described here from the data analytics perspective, led by the author for evaluation methodology and experimental design; production code is not publicly available.</sub>
