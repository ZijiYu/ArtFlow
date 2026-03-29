# ArtFlow

ArtFlow is a closed-loop analysis pipeline for Chinese painting appreciation. It starts from a single painting image, bootstraps a slot-based understanding of the work, iteratively expands evidence through domain-specific reasoning and retrieval, and finally produces a grounded appreciation text rather than a one-shot freeform answer.

The implementation is organized as a layered system:

- `preception_layer/`: image bootstrap, initial slot construction, early grounding, and downstream evidence merge.
- `src/cot_layer/`: slot-oriented CoT execution, thread management, runtime state, and closed-loop coordination.
- `src/cross_validation_layer/`: round-table review for blind-spot detection, follow-up questions, and slot lifecycle checks.
- `src/reflection_layer/`: retrieval planning, routing, convergence judgment, and final appreciation synthesis.
- `../cluster_match/`: companion structured evaluation toolkit for factor extraction and score analysis.

The current pipeline supports:

- slot-level progressive analysis instead of a single monolithic prompt,
- cumulative runtime memory across rounds,
- local RAG plus optional web search routing,
- downstream discovery tasks for missing catalog, author, and contextual evidence,
- final synthesis that explicitly covers key slots and retained facts.

## Quickstart

### 1. Install

```bash
cd /Users/ken/MM/Pipeline/final_version
uv sync
```

If you do not use `uv`, a minimal fallback is:

```bash
cd /Users/ken/MM/Pipeline/final_version
python3 -m venv .venv
source .venv/bin/activate
pip install PyYAML
```

### 2. Configure the API

Edit [`config.yaml`](./config.yaml) and set either:

- `api.key`, or
- `api.key_file` to a local secret file that is not committed.

Optional web search is controlled through `web_search.enabled`. When disabled, the pipeline runs with the local retrieval path only.

### 3. Run a single-pass analysis

```bash
cd /Users/ken/MM/Pipeline/final_version
python pics/main.py \
  --config config.yaml \
  --image /absolute/path/to/painting.jpg
```

This produces a standard slots run with outputs such as `domain_outputs.json`, `cross_validation.json`, `routing.json`, and `final_appreciation_prompt.md`.

### 4. Run the full closed loop

```bash
cd /Users/ken/MM/Pipeline/final_version
python pics/closed_loop.py \
  --config config.yaml \
  --image /absolute/path/to/painting.jpg \
  --text "请对这幅国画做严谨分析。"
```

This triggers the full pipeline:

1. bootstrap in `preception_layer`,
2. slot-level CoT analysis,
3. round-table validation and routing,
4. downstream discovery when evidence is missing,
5. cumulative final appreciation generation.

The run directory is written under the `closed_loop.output_dir` configured in [`config.yaml`](./config.yaml).

### 5. Run tests

```bash
cd /Users/ken/MM/Pipeline/final_version
PYTHONPATH=/Users/ken/MM/Pipeline/final_version pytest tests -q
```

## Outputs

The most important outputs are:

- `perception_bootstrap/slots.jsonl`: initial slot schema,
- `runtime_state/slots_final.jsonl`: final slot state after closed-loop updates,
- `runtime_state/downstream_rag_cache.json`: retrieval cache accumulated across rounds,
- `slots_rounds/.../domain_outputs.json`: per-round slot reasoning outputs,
- `slots_rounds/.../cross_validation.json`: blind spots, follow-ups, and lifecycle review,
- `final_appreciation_prompt.md`: final grounded appreciation text.

## More Detail

For a full architecture and research-oriented explanation, see [`TECHNICAL_REPORT.md`](./TECHNICAL_REPORT.md).
