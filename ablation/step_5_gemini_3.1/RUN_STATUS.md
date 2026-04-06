## step_5_gemini_3.1

- Requested model name: `gemini-3.1-pro-preview`
- Executable model ID on current gateway: `google/gemini-3.1-pro-preview`
- Shared config updated from: `/Users/ken/MM/Pipeline/final_version/config.yaml`
- Input image list snapshot: `/Users/ken/MM/Pipeline/ablation/step_5_gemini_3.1/input_images.jsonl`
- Config snapshot: `/Users/ken/MM/Pipeline/ablation/step_5_gemini_3.1/final_version_config_snapshot.yaml`

### Saved artifacts

- Ablation configs: `/Users/ken/MM/Pipeline/ablation/step_5_gemini_3.1/configs`
- Manifest: `/Users/ken/MM/Pipeline/ablation/step_5_gemini_3.1/ablation_manifest.json`
- API diagnostics:
  - prefixed model test: `/Users/ken/MM/Pipeline/ablation/step_5_gemini_3.1/api_diag_prefixed`
  - unprefixed model test: `/Users/ken/MM/Pipeline/ablation/step_5_gemini_3.1/api_diag_unprefixed`
- Smoke test artifacts:
  - `/Users/ken/MM/Pipeline/ablation/step_5_gemini_3.1/_uv_smoke_test`
  - `/Users/ken/MM/Pipeline/ablation/step_5_gemini_3.1/_uv_smoke_test_prefixed`
- Batch sample artifacts:
  - `/Users/ken/MM/Pipeline/ablation/step_5_gemini_3.1/baseline/batch_runs/1_女史箴图`

### Current verification result

- The bare model name `gemini-3.1-pro-preview` returns gateway `503`.
- The prefixed model ID `google/gemini-3.1-pro-preview` returns `200` and can enter the real pipeline.
- A direct `closed_loop.py` smoke run with the prefixed model completed perception and multiple closed-loop stages before manual interruption.
- The batch runner was updated to support a custom runner command so child processes can use `uv run --with pillow --with openai python`.

### Current batch run

- This run is a model-replacement test only: `baseline` variant, with all modules kept on.
- Detached batch PID: `33209`
- PID file: `/Users/ken/MM/Pipeline/ablation/step_5_gemini_3.1/model_replacement_baseline20.pid`
- Driver log: `/Users/ken/MM/Pipeline/ablation/step_5_gemini_3.1/model_replacement_baseline20.log`
- Current sample output root: `/Users/ken/MM/Pipeline/ablation/step_5_gemini_3.1/baseline/batch_runs`
