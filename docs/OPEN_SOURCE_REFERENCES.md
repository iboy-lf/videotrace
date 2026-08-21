# Open Source References

VideoTrace is implemented as an independent Python project. The repositories below were reviewed for design ideas; source is not copied wholesale. This file records provenance so that architectural borrowing remains explicit in interviews and future releases.

## Hello-Agents course repository

- Repository: `datawhalechina/hello-agents`
- Reviewed commit: `45dd84e626a91997294ac8d4d44f18b29a411c6e`
- License: CC BY-NC-SA 4.0
- Ideas reviewed: Agent framework layering, context engineering, memory and retrieval, and traceable tool execution.
- VideoTrace use: conceptual reference only. The runtime and tests in this repository are independently written.

## HelloAgents framework

- Repository: `jjyaoao/helloagents`
- Reviewed commit: `5432566d01ea1c2095c4a717fe2a010aa1c3b0bd`
- License: CC BY-NC-SA 4.0
- Ideas reviewed: structured tool responses, schema-aware registration, retry handling, circuit breakers, and session persistence.
- VideoTrace use: independently implemented `ToolResponse`, input/output contract checks, bounded retries, per-tool circuit breakers, and serializable traces. The restrictive non-commercial ShareAlike license is why framework code was not copied directly.

## RAGent

- Repository: `nageoffer/ragent`
- Reviewed commit: `5aabe1eaeedae70df353010fc73b315a8a1eef89`
- License: Apache-2.0
- Popularity snapshot: about 3.6k GitHub stars on 2026-08-19.
- Language: Java.
- Ideas reviewed: parallel multi-channel retrieval, stable channel contracts, weighted Reciprocal Rank Fusion, candidate-pool truncation before reranking, query intent, and conversation memory.
- VideoTrace use: the Python retrieval pipeline follows the same high-level separation of retrieval channels, fusion, reranking, and evidence post-processing. Implementations are native to VideoTrace and adapted to timestamped video segments.

## Attribution policy

- Preserve this file when sharing the project.
- Add any future copied or substantially adapted code with file-level attribution and the upstream license notice.
- Do not import CC BY-NC-SA code into a differently licensed distribution without reviewing ShareAlike and non-commercial obligations.
