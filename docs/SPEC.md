# SPEC

## Problem
Long videos are hard to search, verify, and summarize with evidence.

## Product
A multimodal agent that converts a long video into a structured knowledge pack:
- timeline summary
- evidence-backed Q&A
- key segments with timestamps
- clickable source-video evidence windows

## Non-goals
- Training a foundation model from scratch
- Pure data annotation work
- Pure SFT or DPO without a visible product loop
- Public leaderboard or benchmark reproduction

## What makes it strong
- Evidence-first outputs
- Agent planning, not only generation
- Retrieval over multimodal chunks
- Lightweight training on planner/scorer/verifier modules
- Real-video end-to-end validation and replayable evidence

## Primary outputs
1. JSON knowledge pack
2. Markdown report
3. Timestamped timeline
4. Source-video evidence-window manifest
5. Training/model-card and reproducibility artifacts

## Audience
- Internship interviews for VLM, LLM, Agent, and multimodal algorithm roles
- A portfolio project that demonstrates architecture, training, and evaluation
