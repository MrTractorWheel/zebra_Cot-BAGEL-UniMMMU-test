# -*- coding: utf-8 -*-
"""
Uni-MMMU sampling with Bagel-Zebra-CoT.

Runs the six Uni-MMMU tasks (science, math, code, jigsaw, sliding, maze) with
the Bagel-Zebra-CoT unified model, producing outputs in the exact directory
layout expected by Uni-MMMU's eval_ummmu.py:

    <ummmu_root>/outputs/<model_name>/<task>/case_*/...

The task protocols (prompts, few-shot demos, file names, resume markers) are
faithful ports of the official templates in sample_code_example/gpt/, with the
model-specific generate_text/generate_image implemented via bagel_backend.

Fully resumable: finished cases are marked with `_done.ok` and skipped on
re-run, so the benchmark can be split across multiple bounded sessions
(e.g. molab's 12h limit) using --time-budget-hours.

Example:
    python run_sampling.py \
        --checkpoint-dir /path/to/Bagel-Zebra-CoT \
        --bagel-repo /path/to/Bagel-Zebra-CoT-repo \
        --ummmu-root /path/to/Uni-MMMU \
        --task all --time-budget-hours 11
"""

import argparse
import glob
import json
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path
from string import Template
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

# Ensure sibling modules are importable even when the interpreter runs in
# "safe path" mode (e.g. PYTHONSAFEPATH set by the host environment).
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bagel_backend import (
    BagelZebraCoTBackend,
    ContextItem,
    add_image_path,
    add_text,
)

# ======================================================================
# Globals set from CLI in main()
# ======================================================================
BACKEND: Optional[BagelZebraCoTBackend] = None
UMMMU_ROOT: Optional[Path] = None
OUT_BASE: Optional[Path] = None
LIMIT: Optional[int] = None
DEADLINE_TS: Optional[float] = None  # epoch seconds; stop starting new cases after this
SYNC_CMD: Optional[str] = None       # shell command to persist outputs (e.g. git push)
SYNC_INTERVAL_S: float = 20 * 60
_LAST_SYNC_TS: float = 0.0

TASK_ORDER = ["science", "math", "code", "jigsaw", "sliding", "maze"]


def out_of_time() -> bool:
    return DEADLINE_TS is not None and time.time() >= DEADLINE_TS


def maybe_sync(force: bool = False) -> None:
    """Run SYNC_CMD (if configured) at most every SYNC_INTERVAL_S. Non-fatal:
    a failed sync must never kill the sampling run."""
    global _LAST_SYNC_TS
    if not SYNC_CMD:
        return
    now = time.time()
    if not force and now - _LAST_SYNC_TS < SYNC_INTERVAL_S:
        return
    _LAST_SYNC_TS = now
    print(f"[sync] {SYNC_CMD}", flush=True)
    try:
        subprocess.run(SYNC_CMD, shell=True, check=False, timeout=1800)
    except Exception as e:
        print(f"[sync] failed (non-fatal): {e}", flush=True)


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def sanitize_name(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-\.]+", "_", (s or "").strip())[:120] or "item"


def write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def log_case(task: str, case_id: str, status: str, t0: float) -> None:
    print(f"[{task}] {case_id}: {status} ({time.time() - t0:.1f}s)", flush=True)
    maybe_sync()


# ======================================================================
# 1) SCIENCE  (understanding guides generation: text reasoning + 1 image)
# ======================================================================
SCIENCE_PROMPT = """You are a unified vision-language model. You will be given:

(1) one initial image, and 
(2) a textual condition describing an operation/environmental change.

Your job:
- Infer the UNIQUE final state using real-world knowledge and deterministic reasoning.
- Do NOT restate the condition as the result; derive the result causally.
- Do NOT introduce new persistent objects unless they follow necessarily from the condition (e.g., foam from gas, puddle from melting).
- Keep the scene consistent: objects present initially should remain unless the condition implies their removal.
- Output EXACTLY:
<OUTPUT_PROMPT> a concise, deterministic explanation (\u2264120 words) ending with a precise visual description of the final state. No hedging, no multiple possibilities. </OUTPUT_PROMPT>
And generate EXACTLY ONE image depicting the final state (no extra text).

Hard constraints:
- Deterministic, single outcome.
- No meta talk about prompts, models, or pipelines.
- Do not copy the condition as the result; reason from it.
"""


def run_science() -> bool:
    data_json = UMMMU_ROOT / "data" / "science" / "dim_all.json"
    run_root = ensure_dir(OUT_BASE / "science")

    obj = json.loads(data_json.read_text(encoding="utf-8"))
    cases: List[Dict[str, Any]] = []
    for block in obj:
        for s in block.get("samples", []):
            imgs = s.get("input_image_file_path_list") or []
            cond = s.get("input_prompt")
            if imgs and cond and isinstance(imgs, list) and isinstance(cond, str):
                cases.append(
                    {
                        "initial_image": imgs[0],
                        "condition": cond.strip(),
                        "meta": {
                            "level_1": block.get("level_1_category"),
                            "level_2": block.get("level_2_category"),
                        },
                    }
                )
    if LIMIT:
        cases = cases[:LIMIT]

    completed = True
    summary = {"run_root": str(run_root), "count_total": len(cases), "per_item": [],
               "count_processed": 0, "count_success": 0, "count_error": 0, "count_skipped": 0}

    for idx, case in enumerate(tqdm(cases, desc="science"), 1):
        case_dir = ensure_dir(run_root / f"case_{idx:02d}")
        done_marker = case_dir / "_done.ok"
        if done_marker.exists():
            summary["count_skipped"] += 1
            continue
        if out_of_time():
            completed = False
            break

        t0 = time.time()
        initial_image = str(UMMMU_ROOT / case["initial_image"])
        record = {"id": f"case_{idx:02d}", "status": "unknown", "case_dir": str(case_dir),
                  "initial_image": initial_image, "condition": case["condition"],
                  "meta": case["meta"], "text_file": None, "images_saved": [], "errors": []}
        summary["count_processed"] += 1
        try:
            ctx: List[ContextItem] = []
            add_text(ctx, SCIENCE_PROMPT)
            add_text(ctx, "Initial image:")
            add_image_path(ctx, initial_image)
            add_text(ctx, f"Condition: {case['condition']}")

            full_text = BACKEND.generate_text(ctx, max_tokens=512)
            (case_dir / "model_text.txt").write_text(full_text or "", encoding="utf-8")
            record["text_file"] = str(case_dir / "model_text.txt")

            add_text(ctx, full_text)
            img_path = BACKEND.generate_image(ctx, out_path=case_dir / "model_image_01.png")
            record["images_saved"] = [img_path]

            done_marker.write_text("ok", encoding="utf-8")
            record["status"] = "ok"
            summary["count_success"] += 1
        except Exception as e:
            record["status"] = "error"
            record["errors"] = [f"{type(e).__name__}: {e}", traceback.format_exc(limit=2)]
            summary["count_error"] += 1
        write_json(record, case_dir / "result.json")
        summary["per_item"].append(record)
        log_case("science", record["id"], record["status"], t0)

    summary["completed"] = completed
    write_json(summary, run_root / "summary.json")
    return completed


# ======================================================================
# 2) MATH / GEOMETRY  (auxiliary-line overlay image + textual solution)
# ======================================================================
MATH_PROMPT_TMPL = Template("""You are a geometry diagram editor and solver.

TASK ORDER:
1) OVERLAY: On the attached base figure, overlay the auxiliary lines EXACTLY as specified below.
   - Add overlays only; do not move/erase the original objects or labels.
   - Keep labels (A, B, C, \u2026) unchanged and clearly visible.
   - Draw clean, visible lines.

2) REASONING: Give a concise, logically ordered solution or proof (\u2264150 words), using the constructed auxiliary lines.
   - Keep math tokens (\u25b3, \u2220, \u221a, \u03c0, \u00b0) unchanged.
   - Reference elements by their labels.

3) FINISHING:
   - For calculation problems, end with:  **Final answer: <VALUE>**.
   - For proving problems, end with:     **Conclusion: <STATEMENT>**.

PROBLEM:
$PROBLEM_TEXT

CHOICES (if any):
$CHOICES_TEXT

AUXILIARY LINES TO DRAW (English; follow exactly and draw these first):
$AUX_EN
""")


def run_math() -> bool:
    filtered_json = UMMMU_ROOT / "data" / "math_data" / "filtered.json"
    root_dir = filtered_json.parent
    out_dir = ensure_dir(OUT_BASE / "math")

    data: Dict[str, Dict[str, Any]] = json.loads(filtered_json.read_text(encoding="utf-8"))
    items: List[Tuple[str, str, Dict[str, Any]]] = []
    for big_k, group in data.items():
        for small_k, item in group.items():
            items.append((big_k, small_k, item))
    if LIMIT:
        items = items[:LIMIT]

    completed = True
    summary = {"out_dir": str(out_dir), "count_total": len(items), "per_item": [],
               "count_processed": 0, "count_success": 0, "count_error": 0, "count_skipped": 0}

    for big_k, small_k, item in tqdm(items, desc="math"):
        dir_name = f"{sanitize_name(big_k)}__{sanitize_name(small_k)}"
        ex_dir = ensure_dir(out_dir / dir_name)
        done_marker = ex_dir / "_done.ok"
        if done_marker.exists():
            summary["count_skipped"] += 1
            continue
        if out_of_time():
            completed = False
            break

        t0 = time.time()
        record = {"id": dir_name, "big_key": big_k, "small_key": small_k, "status": "unknown",
                  "ex_dir": str(ex_dir), "type": item.get("type"),
                  "original_image": item.get("original_image"),
                  "text_file": None, "images_saved": [], "errors": []}
        summary["count_processed"] += 1

        orig_rel = item.get("original_image")
        orig_abs = (root_dir / orig_rel).resolve() if orig_rel else None
        if not orig_abs or not orig_abs.exists():
            record["status"] = "error"
            record["errors"].append(f"Original image not found: {orig_abs}")
            write_json(record, ex_dir / "result.json")
            summary["count_error"] += 1
            summary["per_item"].append(record)
            continue

        problem_text = item.get("problem_text_en") or item.get("problem_text") or "(no problem text)"
        choices = item.get("choices_en")
        choices_text = "\n".join(choices) if isinstance(choices, list) and choices else "(no choices)"
        aux_en = (item.get("auxiliary_text_en") or "").strip()
        prompt = MATH_PROMPT_TMPL.safe_substitute(
            PROBLEM_TEXT=problem_text, CHOICES_TEXT=choices_text, AUX_EN=aux_en
        )

        try:
            ctx: List[ContextItem] = []
            add_text(ctx, prompt)
            add_text(ctx, "BASE FIGURE: The following image is the original diagram.")
            add_image_path(ctx, str(orig_abs))

            img_path = BACKEND.generate_image(
                ctx,
                out_path=ex_dir / "model_image_01.png",
                prompt_suffix=(
                    "STEP 1 \u2014 OVERLAY now: Output EXACTLY ONE image of the base figure "
                    "with the auxiliary lines overlaid as specified. "
                    "Do not change existing objects/labels, do not add text, no captions."
                ),
            )
            record["images_saved"] = [img_path]

            add_text(ctx, "OVERLAY RESULT (reference for reasoning):")
            add_image_path(ctx, img_path)

            text_out = BACKEND.generate_text(
                ctx,
                prompt_suffix=(
                    "STEP 2 \u2014 REASONING now: Provide ONLY the concise solution/proof (\u2264150 words), "
                    "using the auxiliary lines. Keep math tokens (\u25b3, \u2220, \u221a, \u03c0, \u00b0) unchanged. "
                    "For calculation problems, end with '**Final answer: <VALUE>**'. "
                    "For proving problems, end with '**Conclusion: <STATEMENT>**'. "
                    "Output TEXT ONLY (no images, no markdown images)."
                ),
                max_tokens=768,
            )
            (ex_dir / "model_text.txt").write_text(text_out or "", encoding="utf-8")
            record["text_file"] = str(ex_dir / "model_text.txt")

            done_marker.write_text("ok", encoding="utf-8")
            record["status"] = "ok"
            summary["count_success"] += 1
        except Exception as e:
            record["status"] = "error"
            record["errors"] = [f"{type(e).__name__}: {e}", traceback.format_exc(limit=2)]
            summary["count_error"] += 1
        write_json(record, ex_dir / "result.json")
        summary["per_item"].append(record)
        log_case("math", dir_name, record["status"], t0)

    summary["completed"] = completed
    write_json(summary, out_dir / "summary.json")
    return completed


# ======================================================================
# 3) CODE / SVG RENDERING  (text render-summary + 1 rendered image)
# ======================================================================
CODE_PROMPT = """You will be given SVG source code. Internally parse and render it without tools, then output:
(1) <RENDER_SUMMARY>\u2026</RENDER_SUMMARY> (\u226460 words, objective, deterministic description of the final image)
(2) One final rendered image.
Rendering rules (strict):
Canvas size: determined by <svg> width/height and viewBox.
Background: only as explicitly drawn (e.g. a <rect>); do not add defaults.
Coordinates: respect viewBox; (x,y,r,cx,cy, etc.) in user space.
Stacking: later elements overlay earlier ones.
Styles: fill, stroke, stroke-width, opacity, fill-rule, stroke-linecap/join, etc.
Defaults follow SVG spec (e.g. fill=black, stroke=none).
Transforms: apply from right to left; all path/text positions affected.
On success, output summary + image.
Never output reasoning steps or explanations.
"""


def run_code() -> bool:
    dataset_dir = UMMMU_ROOT / "data" / "svg"
    meta_path = dataset_dir / "metadata.json"
    out_root = ensure_dir(OUT_BASE / "code")

    cases = json.loads(meta_path.read_text(encoding="utf-8")).get("samples", [])
    if LIMIT:
        cases = cases[:LIMIT]

    completed = True
    summary = {"out_root": str(out_root), "count_total": len(cases), "per_item": [],
               "count_processed": 0, "count_success": 0, "count_error": 0, "count_skipped": 0}

    for idx, s in enumerate(tqdm(cases, desc="code"), 1):
        sid = s.get("id", f"noid_{idx:02d}")
        case_dir = ensure_dir(out_root / f"case_{idx:02d}_{sid}")
        done_marker = case_dir / "_done.ok"
        if done_marker.exists():
            summary["count_skipped"] += 1
            continue
        if out_of_time():
            completed = False
            break

        t0 = time.time()
        svg_rel = s.get("svg")
        record = {"id": f"case_{idx:02d}_{sid}", "status": "unknown", "case_dir": str(case_dir),
                  "svg": str(svg_rel), "png": str(s.get("png", "")),
                  "difficulty": s.get("difficulty", ""),
                  "text_file": None, "images_saved": [], "errors": []}
        summary["count_processed"] += 1

        svg_path = dataset_dir / svg_rel if svg_rel else None
        if not svg_path or not svg_path.exists():
            record["status"] = "error"
            record["errors"].append(f"SVG not found: {svg_path}")
            write_json(record, case_dir / "result.json")
            summary["count_error"] += 1
            summary["per_item"].append(record)
            continue

        svg_text = svg_path.read_text(encoding="utf-8")
        try:
            ctx: List[ContextItem] = []
            add_text(ctx, CODE_PROMPT)
            add_text(ctx, svg_text)

            full_text = BACKEND.generate_text(
                ctx, prompt_suffix="now thinking, do not generate image now", max_tokens=512
            )
            (case_dir / "model_text.txt").write_text(full_text or "", encoding="utf-8")
            record["text_file"] = str(case_dir / "model_text.txt")

            ctx2: List[ContextItem] = []
            add_text(ctx2, CODE_PROMPT)
            add_text(ctx2, svg_text)
            add_text(ctx2, full_text)
            img_path = BACKEND.generate_image(
                ctx2,
                out_path=case_dir / "model_image_01.png",
                prompt_suffix="Generate EXACTLY ONE final rendered image of the SVG (no extra text).",
            )
            record["images_saved"] = [img_path]

            done_marker.write_text("ok", encoding="utf-8")
            record["status"] = "ok"
            summary["count_success"] += 1
        except Exception as e:
            record["status"] = "error"
            record["errors"] = [f"{type(e).__name__}: {e}", traceback.format_exc(limit=2)]
            summary["count_error"] += 1
        write_json(record, case_dir / "result.json")
        summary["per_item"].append(record)
        log_case("code", record["id"], record["status"], t0)

    summary["completed"] = completed
    write_json(summary, out_root / "summary.json")
    return completed


# ======================================================================
# 4) JIGSAW  (2 candidate completion images + choice text)
# ======================================================================
JIGSAW_PROMPT = """
You are a unified vision-language model. You will be given:
(1) a 2\u00d72 reference image with the bottom-right cell hidden, and
(2) two candidate patch images (\u201cCandidate 0\u201d and \u201cCandidate 1\u201d).

Your job:
- For each candidate, synthesize a completed 2\u00d72 image by placing that candidate EXACTLY into the bottom-right cell. Keep the other three cells pixel-identical to the reference (no filtering, no re-rendering). If sizes differ, only scale the candidate to fit that quadrant; do NOT rotate, mirror, or alter colors.
- Compare the two completed results and decide which candidate yields the correct completion.

Output EXACTLY the following, in order:

1) A single image with Candidate 0 placed in the bottom-right cell

2) A single image with Candidate 1 placed in the bottom-right cell


3) analysis comparing seam continuity, color/texture gradient, structural alignment, and global semantics

4) One strict JSON object with your decision, wrapped as:
<FINAL_ANSWER_JSON>
{"choice": 0 or 1, "rationale": "\u226430 words decisive cue"}
</FINAL_ANSWER_JSON>

Hard constraints:
- Deterministic, single outcome. No hedging, no multiple possibilities.
- No meta talk about prompts, models, or pipelines.
- Do not restate the task as the answer; reason from visual evidence.
- The only edits allowed are pasting the candidate into the bottom-right cell and necessary size matching for that cell. All other pixels must remain identical to the reference.

Inputs :
"""


def _resolve_data_path(p: Optional[str], dataset_dir: Path) -> Optional[Path]:
    """Resolve a metadata path against the Uni-MMMU root, then the dataset dir."""
    if not p:
        return None
    pth = Path(p)
    if pth.is_absolute():
        return pth
    cand = (UMMMU_ROOT / p).resolve()
    if cand.exists():
        return cand
    return (dataset_dir / p).resolve()


def run_jigsaw() -> bool:
    dataset_dir = UMMMU_ROOT / "data" / "jigsaw_dataset_2x2ref"
    out_root = ensure_dir(OUT_BASE / "jigsaw")

    items = json.loads((dataset_dir / "metadata.json").read_text(encoding="utf-8")).get("items", [])
    if LIMIT:
        items = items[:LIMIT]

    completed = True
    summary = {"out_root": str(out_root), "count_total": len(items), "per_item": [],
               "count_processed": 0, "count_success": 0, "count_error": 0, "count_skipped": 0}

    for idx, it in enumerate(tqdm(items, desc="jigsaw")):
        ex_id = it.get("id", f"ex_{idx:05d}")
        ex_dir = ensure_dir(out_root / sanitize_name(ex_id))
        done_marker = ex_dir / "_done.ok"
        if done_marker.exists():
            summary["count_skipped"] += 1
            continue
        if out_of_time():
            completed = False
            break

        t0 = time.time()
        cands = it.get("candidate_paths") or ["", ""]
        record = {"id": ex_id, "dataset_index": idx, "status": "unknown", "ex_dir": str(ex_dir),
                  "ref_2x2": it.get("ref_panel", {}).get("ref_image_path"),
                  "cand0": cands[0], "cand1": cands[1],
                  "text_file": None, "images_saved": [], "errors": []}
        summary["count_processed"] += 1

        try:
            ref_path = _resolve_data_path(record["ref_2x2"], dataset_dir)
            cand0_path = _resolve_data_path(record["cand0"], dataset_dir)
            cand1_path = _resolve_data_path(record["cand1"], dataset_dir)
            for lbl, p in [("ref_2x2", ref_path), ("cand0", cand0_path), ("cand1", cand1_path)]:
                if not p or not p.exists():
                    raise FileNotFoundError(f"Missing input {lbl}: {p}")

            base_ctx: List[ContextItem] = []
            add_text(base_ctx, JIGSAW_PROMPT)
            add_text(base_ctx, "REFERENCE_2x2:")
            add_image_path(base_ctx, str(ref_path))
            add_text(base_ctx, "CANDIDATE_0:")
            add_image_path(base_ctx, str(cand0_path))
            add_text(base_ctx, "CANDIDATE_1:")
            add_image_path(base_ctx, str(cand1_path))

            img0_path = BACKEND.generate_image(
                list(base_ctx),
                out_path=ex_dir / "model_image_01.png",
                prompt_suffix="Output ONLY item (1): a single image with Candidate 0 placed in the bottom-right cell. No text.",
            )
            add_text(base_ctx, "COMPLETED WITH CANDIDATE 0:")
            add_image_path(base_ctx, img0_path)

            img1_path = BACKEND.generate_image(
                list(base_ctx),
                out_path=ex_dir / "model_image_02.png",
                prompt_suffix="Output ONLY item (2): a single image with Candidate 1 placed in the bottom-right cell. No text.",
            )
            record["images_saved"] = [img0_path, img1_path]

            add_text(base_ctx, "COMPLETED WITH CANDIDATE 1:")
            add_image_path(base_ctx, img1_path)

            text_out = BACKEND.generate_text(
                list(base_ctx),
                prompt_suffix=(
                    'Now output EXACTLY ONE <FINAL_ANSWER_JSON>{"choice": 0 or 1, '
                    '"rationale": "\u226430 words"}</FINAL_ANSWER_JSON>\n'
                    "Do not output any additional images."
                ),
                max_tokens=512,
            )
            (ex_dir / "model_text.txt").write_text(text_out or "", encoding="utf-8")
            record["text_file"] = str(ex_dir / "model_text.txt")

            done_marker.write_text("ok", encoding="utf-8")
            record["status"] = "ok"
            summary["count_success"] += 1
        except Exception as e:
            record["status"] = "error"
            record["errors"] = [f"{type(e).__name__}: {e}", traceback.format_exc(limit=2)]
            summary["count_error"] += 1
        write_json(record, ex_dir / "result.json")
        summary["per_item"].append(record)
        log_case("jigsaw", ex_id, record["status"], t0)

    summary["completed"] = completed
    write_json(summary, out_root / "summary.json")
    return completed


# ======================================================================
# 5) SLIDING PUZZLE  (k step images + final move list)
# ======================================================================
SLIDING_PROMPT = """You are a precise sliding puzzle solver.

TASK
- You will be given two images: an INITIAL state and a FINAL state of a 3x3 sliding puzzle.
- The goal is to find the sequence of moves to transform the INITIAL state into the FINAL state.

SEMANTICS
- The puzzle is a 3x3 grid with 8 colored tiles and one empty space.
- The RED square represents the EMPTY space.
- A "move" consists of sliding an adjacent colored tile INTO the empty (red) space.
- Moves are named by the direction the empty (red) space moves. For example, if the blue tile is directly above the red space, moving the blue tile down into the red space's position is a "up" move.
- Legal moves: up, down, left, right only. One tile per step.

OUTPUT FORMAT (STRICT)
1) MULTI-IMAGE MODE \u2014 generate a SEQUENCE OF SEPARATE IMAGES, one per move:
   - Each output image must depict the puzzle state AFTER applying exactly one legal move.
   - Do NOT include the initial (pre-move) state.
   - Keep the visual style identical to the inputs; only tile positions change.
   - The number of returned images MUST equal the number of moves in the final answer (see step 2).
   - Absolutely FORBIDDEN: collage/montage/grid/stacked images; no arrows, captions, or overlays; no GIFs/animations/video.

2) After all step images, emit EXACTLY ONE LINE containing ONLY the final move list as a JSON array of lowercase strings, wrapped as:
   <ANSWER_JSON>["down","right","up"]</ANSWER_JSON>


NO EXTRAS
- No tools, no explanations, and no text other than the single <ANSWER_JSON>\u2026</ANSWER_JSON> line.
- Do not restate the instructions.

REMINDERS
- First decide the full path, then emit the image sequence (one image per move), then the single <ANSWER_JSON> line.
- One move per image; images must be separate files/parts, not stitched.

After the single <ANSWER_JSON>\u2026</ANSWER_JSON> line, output nothing else.
"""

SLIDING_STEP_GLOB = "demo_step_*.png"


def _sliding_fewshot_ctx() -> List[ContextItem]:
    d = UMMMU_ROOT / "data" / "sliding"
    problem_imgs = [d / "demo_3x3_00001_steps" / "demo_step_0000.png",
                    d / "demo_3x3_00001_steps" / "demo_step_0003.png"]
    solution_imgs = [d / "demo_3x3_00001_steps" / "demo_step_0001.png",
                     d / "demo_3x3_00001_steps" / "demo_step_0002.png"]
    ans_json = d / "demo_3x3_steps_words_00001.json"

    ctx: List[ContextItem] = []
    add_text(ctx, "--- DEMONSTRATION START ---")
    if problem_imgs[0].exists():
        add_text(ctx, "DEMONSTRATION: The initial state.")
        add_image_path(ctx, str(problem_imgs[0]))
    if problem_imgs[1].exists():
        add_text(ctx, "DEMONSTRATION: The final state to reach.")
        add_image_path(ctx, str(problem_imgs[1]))
    add_text(ctx, "DEMONSTRATION: The sequence of moves to solve the puzzle (one image per move).")
    for p in solution_imgs:
        if p.exists():
            add_image_path(ctx, str(p))
    try:
        obj = json.loads(ans_json.read_text(encoding="utf-8"))
        seq = [str(s).lower() for s in obj.get("steps_words", [])]
        add_text(ctx, "DEMONSTRATION: The final moves list in JSON format.")
        add_text(ctx, f"<ANSWER_JSON>{json.dumps(seq)}</ANSWER_JSON>")
    except Exception:
        pass
    add_text(ctx, "--- DEMONSTRATION END ---")
    return ctx


def run_sliding() -> bool:
    summary_json = UMMMU_ROOT / "data" / "sliding" / "summary_steps_le_8.json"
    run_root = ensure_dir(OUT_BASE / "sliding")

    records_full = json.loads(summary_json.read_text(encoding="utf-8")).get("items", [])
    if LIMIT:
        records_full = records_full[:LIMIT]

    completed = True
    summary = {"run_root": str(run_root), "count_total": len(records_full), "per_item": [],
               "count_processed": 0, "count_success": 0, "count_error": 0, "count_skipped": 0}
    fewshot_ctx = _sliding_fewshot_ctx()

    for rec in tqdm(records_full, desc="sliding"):
        steps_dir = UMMMU_ROOT / rec.get("steps_dir", "")
        if not steps_dir.exists():
            continue
        frames = sorted(steps_dir.glob(SLIDING_STEP_GLOB))
        if len(frames) < 2:
            continue
        init_png = str(frames[0])
        case_name = steps_dir.name

        gt_json = rec.get("steps_words_json")
        gt_path = (UMMMU_ROOT / gt_json) if gt_json else None
        moves: List[str] = []
        if gt_path and gt_path.exists():
            try:
                obj = json.loads(gt_path.read_text(encoding="utf-8"))
                moves = [str(x).lower() for x in obj.get("steps_words", [])]
            except Exception:
                pass
        k = len(moves) if moves else max(0, len(frames) - 1)

        case_dir = ensure_dir(run_root / f"case_{case_name}")
        done_marker = case_dir / "_done.ok"
        if done_marker.exists():
            summary["count_skipped"] += 1
            continue
        if out_of_time():
            completed = False
            break

        t0 = time.time()
        record = {"id": case_name, "status": "unknown", "case_dir": str(case_dir),
                  "init_png": init_png, "k": k, "text_file": None,
                  "images_saved_flatten": [], "errors": []}
        summary["count_processed"] += 1

        try:
            ctx: List[ContextItem] = []
            ctx.extend(fewshot_ctx)
            add_text(ctx, SLIDING_PROMPT)
            add_text(ctx, "Now solve the NEW TASK below. Emit ONE separate image per move, then a single <ANSWER_JSON> line.")
            add_text(ctx, "NEW TASK: Initial state.")
            add_image_path(ctx, init_png)
            add_text(ctx, "NEW TASK: Final state is exactly the same as example")

            cand_dir = ensure_dir(case_dir / "cand_01")
            images_flat: List[str] = []
            stem = Path(init_png).stem
            for i in range(1, k + 1):
                step_text = BACKEND.generate_text(
                    ctx,
                    prompt_suffix=f'Now planing for step {i}, Please output a sentence in the form: "Next, move one step up/down/left/right.',
                    max_tokens=128,
                )
                add_text(ctx, step_text)
                img_path = BACKEND.generate_image(
                    ctx,
                    out_path=cand_dir / f"{stem}_step_{i:04d}.png",
                    prompt_suffix=f"Now, generate the image for step {i}. ",
                )
                images_flat.append(img_path)
                add_image_path(ctx, img_path)

            final_text = BACKEND.generate_text(
                ctx,
                prompt_suffix="Now, emit EXACTLY ONE LINE containing ONLY the final move list "
                              "as <ANSWER_JSON>[...]</ANSWER_JSON>. No other text.",
                max_tokens=256,
            )
            (case_dir / "model_text.txt").write_text(final_text or "", encoding="utf-8")
            record["text_file"] = str(case_dir / "model_text.txt")
            record["images_saved_flatten"] = images_flat

            done_marker.write_text("ok", encoding="utf-8")
            record["status"] = "ok"
            summary["count_success"] += 1
        except Exception as e:
            record["status"] = "error"
            record["errors"] = [f"{type(e).__name__}: {e}", traceback.format_exc(limit=2)]
            summary["count_error"] += 1
        write_json(record, case_dir / "result.json")
        summary["per_item"].append(record)
        log_case("sliding", case_name, f"{record['status']} (k={k})", t0)

    summary["completed"] = completed
    write_json(summary, run_root / "summary.json")
    return completed


# ======================================================================
# 6) MAZE  (k step images + final move list)
# ======================================================================
MAZE_PROMPT = """You are a precise maze solver.

SEMANTICS (for all mazes)
- Black squares: walls (impassable)
- White squares: path (walkable)
- Blue dot: start (the agent)
- Green rectangular frame: goal (reaching any white cell inside the green frame counts as success)
- Legal moves: up, down, left, right only. One cell per step; no diagonals, no jumps; never cross walls.

OUTPUT FORMAT (STRICT)
1) MULTI-IMAGE MODE \u2014 generate a SEQUENCE OF SEPARATE IMAGES, one per move:
   - Each output image must depict the maze state AFTER applying exactly one legal move.
   - Do NOT include the initial (pre-move) state.
   - Keep palette/layout/scale identical to the input; only the blue dot moves.
   - The number of returned images MUST equal the number of moves in the final answer (see step 2).
   - Absolutely FORBIDDEN: any collage/montage/spritesheet/grid/multi-panel/side-by-side/stacked images; no arrows, captions, or overlays; no GIFs/animations/video.

2) After all step images, emit EXACTLY ONE LINE containing ONLY the final move list as a JSON array of lowercase strings, wrapped as:
   <ANSWER_JSON>["right","down","left"]</ANSWER_JSON>


NO EXTRAS
- No tools, no OCR, no explanations, and no text other than the single <ANSWER_JSON>\u2026</ANSWER_JSON> line.
- Do not restate the instructions or the condition.

REMINDERS
- Decide the full path first, then emit the image sequence (one image per move), then the single <ANSWER_JSON> line.
- One move per image; images must be separate files/parts, not stitched together in any way.
"""

MAZE_STEP0 = "maze_step_0000.png"
RE_MAZE_STEPS_DIR = re.compile(r"^(?P<prefix>maze_(?P<h>\d+)x(?P<w>\d+))_(?P<id>\d{5})_steps$")
MAZE_EXAMPLE_ID = "00015"


def _maze_gt_moves(maze_root: Path, step0: Path) -> Tuple[List[str], int]:
    m = RE_MAZE_STEPS_DIR.match(step0.parent.name)
    if m:
        js = maze_root / f"{m.group('prefix')}_steps_{m.group('id')}.json"
        if js.exists():
            try:
                obj = json.loads(js.read_text(encoding="utf-8"))
                if isinstance(obj, dict):
                    moves = obj.get("steps_long") or obj.get("steps") or []
                else:
                    moves = obj
                moves = [str(x).lower() for x in moves]
                if moves:
                    return moves, len(moves)
            except Exception:
                pass
    pngs = sorted(glob.glob(str(step0.parent / "maze_step_*.png")))
    return [], max(0, len(pngs) - 1)


def _maze_fewshot_ctx(maze_root: Path) -> List[ContextItem]:
    ex_dir = maze_root / f"maze_6x6_{MAZE_EXAMPLE_ID}_steps"
    ctx: List[ContextItem] = []
    problem = ex_dir / MAZE_STEP0
    if problem.exists():
        add_image_path(ctx, str(problem))
        add_text(ctx, "DEMONSTRATION: Example problem image above.")
    for i in (1, 2):
        p = ex_dir / f"maze_step_{i:04d}.png"
        if p.exists():
            add_image_path(ctx, str(p))
            add_text(ctx, f"DEMONSTRATION: example step image #{i}.")
    js = maze_root / f"maze_6x6_steps_{MAZE_EXAMPLE_ID}.json"
    try:
        obj = json.loads(js.read_text(encoding="utf-8"))
        seq = [s.lower() for s in (obj.get("steps_long", []) if isinstance(obj, dict) else obj)]
        add_text(ctx, f"DEMONSTRATION: final moves\n<ANSWER_JSON>{json.dumps(seq)}</ANSWER_JSON>")
    except Exception:
        pass
    return ctx


def run_maze() -> bool:
    maze_root = UMMMU_ROOT / "data" / "maze"
    run_root = ensure_dir(OUT_BASE / "maze")

    step0_list = sorted(maze_root.rglob(f"*/{MAZE_STEP0}"))
    step0_list = [
        p for p in step0_list
        if (RE_MAZE_STEPS_DIR.match(p.parent.name) or None)
        and RE_MAZE_STEPS_DIR.match(p.parent.name).group("id") != MAZE_EXAMPLE_ID
    ]
    if LIMIT:
        step0_list = step0_list[:LIMIT]

    completed = True
    summary = {"run_root": str(run_root), "count_total": len(step0_list), "per_item": [],
               "count_processed": 0, "count_success": 0, "count_error": 0, "count_skipped": 0}
    fewshot_ctx = _maze_fewshot_ctx(maze_root)

    for step0 in tqdm(step0_list, desc="maze"):
        mid = RE_MAZE_STEPS_DIR.match(step0.parent.name).group("id")
        case_dir = ensure_dir(run_root / f"case_{mid}")
        done_marker = case_dir / "_done.ok"
        if done_marker.exists():
            summary["count_skipped"] += 1
            continue
        if out_of_time():
            completed = False
            break

        t0 = time.time()
        record = {"id": mid, "status": "unknown", "case_dir": str(case_dir),
                  "step0": str(step0), "text_file": None,
                  "images_saved_flatten": [], "errors": []}
        summary["count_processed"] += 1

        try:
            _, k = _maze_gt_moves(maze_root, step0)

            ctx: List[ContextItem] = []
            ctx.extend(fewshot_ctx)
            add_text(ctx, MAZE_PROMPT)
            add_image_path(ctx, str(step0))

            cand_dir = ensure_dir(case_dir / "cand_01")
            images_flat: List[str] = []
            stem = step0.stem  # maze_step_0000
            for i in range(1, k + 1):
                step_text = BACKEND.generate_text(
                    ctx,
                    prompt_suffix=f'Now planing for step {i}, Please output a sentence in the form: "Next, move one step up/down/left/right.',
                    max_tokens=128,
                )
                add_text(ctx, step_text)
                img_path = BACKEND.generate_image(
                    ctx,
                    out_path=cand_dir / f"{stem}_step_{i:04d}.png",
                    prompt_suffix=f"Now, generate the image for step {i}. ",
                )
                images_flat.append(img_path)
                add_image_path(ctx, img_path)

            final_text = BACKEND.generate_text(
                ctx,
                prompt_suffix="After the images, emit EXACTLY ONE LINE containing ONLY the final move list "
                              "as <ANSWER_JSON>[...]</ANSWER_JSON>. No other text.",
                max_tokens=256,
            )
            (case_dir / "model_text.txt").write_text(final_text or "", encoding="utf-8")
            record["text_file"] = str(case_dir / "model_text.txt")
            record["images_saved_flatten"] = images_flat

            done_marker.write_text("ok", encoding="utf-8")
            record["status"] = "ok"
            summary["count_success"] += 1
        except Exception as e:
            record["status"] = "error"
            record["errors"] = [f"{type(e).__name__}: {e}", traceback.format_exc(limit=2)]
            summary["count_error"] += 1
        write_json(record, case_dir / "result.json")
        summary["per_item"].append(record)
        log_case("maze", f"case_{mid}", record["status"], t0)

    summary["completed"] = completed
    write_json(summary, run_root / "summary.json")
    return completed


# ======================================================================
# Main
# ======================================================================
TASK_FUNCS = {
    "science": run_science,
    "math": run_math,
    "code": run_code,
    "jigsaw": run_jigsaw,
    "sliding": run_sliding,
    "maze": run_maze,
}


def main():
    global BACKEND, UMMMU_ROOT, OUT_BASE, LIMIT, DEADLINE_TS, SYNC_CMD, SYNC_INTERVAL_S

    ap = argparse.ArgumentParser(description="Uni-MMMU sampling with Bagel-Zebra-CoT")
    ap.add_argument("--checkpoint-dir", required=True,
                    help="Path to the downloaded Bagel-Zebra-CoT HF checkpoint folder")
    ap.add_argument("--bagel-repo", required=True,
                    help="Path to the cloned Bagel-Zebra-CoT GitHub repo (for imports)")
    ap.add_argument("--ummmu-root", required=True,
                    help="Path to the Uni-MMMU repo root (contains data/ after extracting data.tar)")
    ap.add_argument("--model-name", default="bagel-zebra-cot",
                    help="Output folder name: <ummmu-root>/outputs/<model-name>")
    ap.add_argument("--task", default="all",
                    help=f"Comma-separated tasks from {TASK_ORDER}, or 'all'")
    ap.add_argument("--limit", type=int, default=None,
                    help="Only run the first N cases of each task (smoke test)")
    ap.add_argument("--num-timesteps", type=int, default=50,
                    help="Diffusion steps per generated image (50=paper default, 24-30=~2x faster)")
    ap.add_argument("--text-temperature", type=float, default=0.3)
    ap.add_argument("--max-mem-gib", type=int, default=90,
                    help="Per-GPU memory cap for device mapping")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--time-budget-hours", type=float, default=None,
                    help="Stop starting new cases after this many hours (leave margin before "
                         "the session limit; finished cases are resumed next run)")
    ap.add_argument("--sync-cmd", default=None,
                    help="Shell command run periodically (and at the end) to persist outputs, "
                         "e.g. 'bash outputs_git.sh push'. Failures are non-fatal.")
    ap.add_argument("--sync-interval-min", type=float, default=20.0,
                    help="Minimum minutes between --sync-cmd invocations")
    args = ap.parse_args()

    UMMMU_ROOT = Path(args.ummmu_root).resolve()
    OUT_BASE = ensure_dir(UMMMU_ROOT / "outputs" / args.model_name)
    LIMIT = args.limit
    if args.time_budget_hours:
        DEADLINE_TS = time.time() + args.time_budget_hours * 3600
    SYNC_CMD = args.sync_cmd
    SYNC_INTERVAL_S = max(1.0, args.sync_interval_min) * 60
    if SYNC_CMD:
        print(f"[main] output sync every >={args.sync_interval_min:g} min: {SYNC_CMD}")

    tasks = TASK_ORDER if args.task.strip().lower() == "all" else [
        t.strip().lower() for t in args.task.split(",") if t.strip()
    ]
    for t in tasks:
        if t not in TASK_FUNCS:
            raise SystemExit(f"Unknown task '{t}'. Valid: {TASK_ORDER}")

    assert (UMMMU_ROOT / "data").exists(), (
        f"{UMMMU_ROOT}/data not found - did you extract data.tar from "
        "Vchitect/Uni-MMMU-Eval into the Uni-MMMU repo?"
    )

    print(f"[main] tasks: {tasks}")
    print(f"[main] outputs -> {OUT_BASE}")
    if DEADLINE_TS:
        print(f"[main] time budget: {args.time_budget_hours}h")

    BACKEND = BagelZebraCoTBackend(
        checkpoint_dir=args.checkpoint_dir,
        bagel_repo=args.bagel_repo,
        max_mem_gib=args.max_mem_gib,
        num_timesteps=args.num_timesteps,
        text_temperature=args.text_temperature,
        seed=args.seed,
    )

    results = {}
    for t in tasks:
        if out_of_time():
            print(f"[main] time budget exhausted before task '{t}' - stopping. "
                  "Re-run the same command in the next session to resume.")
            results[t] = "not_started"
            continue
        print(f"\n===== TASK: {t} =====")
        done = TASK_FUNCS[t]()
        results[t] = "completed" if done else "partial (resume next session)"

    print("\n=== RUN SUMMARY ===")
    for t, s in results.items():
        print(f"  {t}: {s}")
    write_json(results, OUT_BASE / "run_status.json")
    maybe_sync(force=True)


if __name__ == "__main__":
    main()
