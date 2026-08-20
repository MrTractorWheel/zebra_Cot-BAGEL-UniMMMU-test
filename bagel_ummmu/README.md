# Evaluating Bagel-Zebra-CoT on Uni-MMMU (molab, 1x RTX 6000 Pro Blackwell 96GB)

This folder contains everything needed to benchmark
[multimodal-reasoning-lab/Bagel-Zebra-CoT](https://huggingface.co/multimodal-reasoning-lab/Bagel-Zebra-CoT)
on the [Uni-MMMU](https://github.com/Vchitect/Uni-MMMU) benchmark.

## How it works

Uni-MMMU has two stages:

1. **Sampling** — the model under test generates interleaved text + images for
   ~730 cases across 6 tasks (maze, sliding puzzle, jigsaw, geometry, science,
   SVG code rendering). The official `sample_code_example/gpt/*.py` scripts are
   *templates*: they define the prompts, few-shot demos and output layout, but
   leave `generate_text_from_context` / `generate_image_from_context`
   unimplemented — each model plugs in its own backend.
2. **Evaluation** — `eval_ummmu.py` scores the files under
   `outputs/<model_name>/` using two LLM judges (Qwen2.5-VL-72B + Qwen3-32B)
   plus rule-based checks, and writes an Excel summary.

Files here:

| File | Purpose |
|---|---|
| `bagel_backend.py` | Loads Bagel-Zebra-CoT (port of `infz_bf16.py`) and implements the two generation primitives via `InterleaveInferencer` |
| `run_sampling.py` | All 6 task loops (faithful ports of the official templates), resumable, with a session time budget |
| `setup_molab.sh` | One-time setup: repos, Blackwell-compatible PyTorch, flash-attn, checkpoint (~30 GB), dataset |
| `run_sampling.sh` | Sampling launcher with logging |
| `run_eval.sh` | Evaluation launcher (patches the eval script for a single 96 GB GPU) |
| `requirements_molab.txt` | Python deps (torch installed separately for sm_120) |

## Quick start on molab

Upload this `bagel_ummmu/` folder to your molab instance, then:

```bash
# Point WORK_DIR at persistent storage so downloads/outputs survive session restarts!
export WORK_DIR=$HOME/bagel_ummmu

# --- Session 0 (setup + smoke test, ~1-2h incl. downloads) ---
bash setup_molab.sh
bash run_sampling.sh --task science --limit 2      # smoke test: check outputs look sane

# --- Session 1 (sampling) ---
bash run_sampling.sh --task all --time-budget-hours 11

# --- Session 2 (sampling, resumes automatically; repeat until "completed") ---
bash run_sampling.sh --task all --time-budget-hours 11

# --- Final session (evaluation with quantized judges) ---
bash run_eval.sh
```

Results land in `$WORK_DIR/Uni-MMMU/eval/bagel-zebra-cot/all_tasks_summary_bagel-zebra-cot.xlsx`.

## Will it fit in 12-hour sessions?

**VRAM: yes, comfortably.** The model is ~14B params (7B active, MoT) ≈ 30 GB
in bf16 + VAE + activations — well within 96 GB. The judges in the eval stage
only fit as AWQ-quantized variants (~41 GB + ~19 GB), which `run_eval.sh`
configures automatically.

**Wall-clock: not in a single session at default quality — plan 2–3 sampling
sessions.** The workload is ~2,170 generated images + ~2,250 text generations:

| Task | Cases | Images to generate |
|---|---|---|
| maze | 149 | ~870 (one per GT step, avg ~5.8) |
| sliding | 84 | ~500 (one per GT step, avg 6) |
| jigsaw | 150 | 300 (2 per case) |
| code (SVG) | 200 | 200 |
| science | 157 | 157 |
| math | 140 | 140 |

At the default 50 diffusion timesteps expect very roughly ~20–30 s per image
and a few seconds per text call on this GPU → **~15–20 h of sampling total**.
The runner is built for this:

- Every finished case writes a `_done.ok` marker; re-running the same command
  skips finished cases and continues where it left off.
- `--time-budget-hours 11` stops *starting* new cases 11 h in, so the session
  ends cleanly with a valid partial state.
- Tasks run cheapest-first (`science → math → code → jigsaw → sliding → maze`),
  so early sessions finish whole tasks.

If you want to squeeze into fewer sessions, add `--num-timesteps 24`
(~2x faster image generation, slightly lower image quality — note it as a
deviation from the authors' recommended 50). Measure your actual per-image
time from the smoke test and extrapolate before committing.

The evaluation stage is another few hours (judge inference over all cases) —
run it as its own session via `run_eval.sh`.

## Blackwell (sm_120) notes

- The Bagel repo pins `torch==2.5.1`, which **does not support** the
  RTX 6000 Pro Blackwell. `setup_molab.sh` installs current torch from the
  cu128 wheel index instead.
- `flash-attn` is required by BAGEL's attention code. If no prebuilt wheel
  matches, it compiles from source for `sm_120` (30–60 min, one-time; done in
  `setup_molab.sh`).

## Knobs / troubleshooting

- `--task maze,sliding` — run specific tasks only (comma-separated).
- `--limit N` — first N cases per task (smoke tests).
- `--model-name` — output folder name; must match `MODEL_NAME` in `run_eval.sh`.
- Re-running after an error is always safe (resume markers).
- To force a case to re-run, delete its `case_*/` folder (or just `_done.ok`).
- If molab storage is ephemeral, archive progress before the session ends:
  `tar -cf outputs.tar -C $WORK_DIR/Uni-MMMU outputs` and download it; extract
  it back at the start of the next session.

## Caveats for interpreting scores

- The exact prompts/protocol match the official Uni-MMMU GPT sample templates,
  so numbers are comparable to the paper's protocol.
- AWQ-quantized judges (single-GPU constraint) may shift judge-based metrics
  slightly vs. the official bf16 judges. Rule-based metrics (maze/sliding/
  jigsaw image checks) are unaffected. For paper-grade numbers, re-run
  `run_eval.sh` with `USE_AWQ_JUDGES=0` on a multi-GPU machine — the sampling
  outputs are reusable as-is.
- Bagel-Zebra-CoT is trained to reason in its own interleaved format
  ("THOUGHT ... Final Answer: ..."); it may not always comply with Uni-MMMU's
  strict output tags (`<ANSWER_JSON>`, `<FINAL_ANSWER_JSON>`). That
  non-compliance is part of what the benchmark measures.
