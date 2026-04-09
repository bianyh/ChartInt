import argparse
import asyncio
from pathlib import Path

from extra_experiment_utils import (
    ensure_dir,
    image_ssim,
    snapshot_chart_states,
    text_similarity,
    write_json,
)


ROOT = Path("/mnt/data_disk/bianyuhan/chartint_re/chart2code")
REF_ROOT = ROOT / "eval_data" / "data_interaction"
GEN_ROOT = ROOT / "eval_result" / "gpt-5.4" / "data_interaction"
OUT_ROOT = ROOT / "extra_experiments" / "multistate_ablation"
SAMPLE_ORDER_FILE = ROOT / "data_interaction" / "results_summary.json"


ALLOWED_TYPES = {"bar", "line", "scatter"}


def compute_state_indices(data_len: int) -> list[int]:
    anchors = [0, 1, data_len // 2, data_len - 2, data_len - 1]
    deduped = []
    for idx in anchors:
        if idx not in deduped:
            deduped.append(idx)
    return deduped


async def probe_reference(sample_id: str, out_root: Path) -> dict | None:
    ref_html = REF_ROOT / sample_id / "original_code.html"
    if not ref_html.exists():
        return None

    ref_out = out_root / sample_id / "reference"
    ref_result = await snapshot_chart_states(str(ref_html), [0], ref_out)
    if not ref_result.get("ok"):
        return None

    meta = ref_result["meta"]
    data_len = int(meta.get("dataLen", 0))
    if data_len < 5:
        return None

    states = compute_state_indices(data_len)
    if len(states) < 5:
        return None

    return {
        "sample_id": sample_id,
        "data_len": data_len,
        "usable_series_index": meta.get("usableSeriesIndex", 0),
        "states": states,
    }


async def evaluate_sample(sample_id: str, states: list[int], out_root: Path) -> dict:
    ref_html = REF_ROOT / sample_id / "original_code.html"
    gen_html = GEN_ROOT / sample_id / "gen.html"

    ref_out = out_root / sample_id / "reference"
    gen_out = out_root / sample_id / "generated"

    ref_result = await snapshot_chart_states(str(ref_html), states, ref_out)
    gen_result = await snapshot_chart_states(str(gen_html), states, gen_out)

    state_rows = []
    ref_states = {item["data_index"]: item for item in ref_result.get("states", [])}
    gen_states = {item["data_index"]: item for item in gen_result.get("states", [])}
    for state in states:
        ref_state = ref_states.get(
            state,
            {
                "tooltip_text": "",
                "screenshot_path": "",
            },
        )
        gen_state = gen_states.get(
            state,
            {
                "tooltip_text": "",
                "screenshot_path": "",
            },
        )
        sim = text_similarity(ref_state["tooltip_text"], gen_state["tooltip_text"])
        ssim = 0.0
        if ref_state["screenshot_path"] and gen_state["screenshot_path"]:
            ssim = image_ssim(ref_state["screenshot_path"], gen_state["screenshot_path"])
        state_rows.append(
            {
                "data_index": state,
                "reference_tooltip": ref_state["tooltip_text"],
                "generated_tooltip": gen_state["tooltip_text"],
                "tooltip_similarity": round(sim, 4),
                "image_ssim": round(ssim, 4),
            }
        )

    return {
        "sample_id": sample_id,
        "states": state_rows,
        "reference_meta": ref_result.get("meta", {}),
        "generated_meta": gen_result.get("meta", {}),
    }


async def evaluate_core_states(sample_id: str, states: list[int], out_root: Path) -> list[float]:
    record = await evaluate_sample(sample_id, states[:2], out_root)
    return [state["tooltip_similarity"] for state in record["states"]]


def summarize_threshold(results: list[dict], threshold: float) -> dict:
    sample_rows = []
    for record in results:
        pass_flags = [state["tooltip_similarity"] >= threshold for state in record["states"]]
        sample_rows.append({"sample_id": record["sample_id"], "pass_flags": pass_flags})

    total = len(sample_rows)
    fail_at = {}
    delta_at = {}
    previous = 0.0
    for k in [1, 2, 3, 5]:
        ratio = sum(1 for row in sample_rows if not all(row["pass_flags"][:k])) / total
        ratio = round(ratio, 4)
        fail_at[k] = ratio
        delta_at[k] = round(ratio - previous, 4) if k != 1 else None
        previous = ratio

    first_two_both_fail = [row for row in sample_rows if len(row["pass_flags"]) >= 2 and not row["pass_flags"][0] and not row["pass_flags"][1]]
    first_two_both_pass = [row for row in sample_rows if len(row["pass_flags"]) >= 2 and row["pass_flags"][0] and row["pass_flags"][1]]

    later_all_fail_given_core_fail = (
        round(
            sum(1 for row in first_two_both_fail if not any(row["pass_flags"][2:])) / len(first_two_both_fail),
            4,
        )
        if first_two_both_fail
        else None
    )
    later_all_pass_given_core_pass = (
        round(
            sum(1 for row in first_two_both_pass if all(row["pass_flags"][2:])) / len(first_two_both_pass),
            4,
        )
        if first_two_both_pass
        else None
    )

    return {
        "threshold": threshold,
        "total_samples": total,
        "fail_at": fail_at,
        "delta_at": delta_at,
        "first_two_both_fail_count": len(first_two_both_fail),
        "first_two_both_pass_count": len(first_two_both_pass),
        "later_all_fail_given_core_fail": later_all_fail_given_core_fail,
        "later_all_pass_given_core_pass": later_all_pass_given_core_pass,
        "samples": sample_rows,
    }


async def main_async(limit: int) -> None:
    out_root = ensure_dir(OUT_ROOT)
    sample_order = []
    if SAMPLE_ORDER_FILE.exists():
        import json

        with open(SAMPLE_ORDER_FILE, "r", encoding="utf-8") as file:
            sample_order = [str(item["sample"]) for item in json.load(file)]
    else:
        sample_order = [path.name for path in sorted(GEN_ROOT.iterdir(), key=lambda path: int(path.name)) if path.is_dir()]

    selected: list[dict] = []
    early_pass: list[dict] = []
    early_fail: list[dict] = []
    for sample_id in sample_order:
        sample_dir = GEN_ROOT / sample_id
        if not sample_dir.is_dir():
            continue
        probe = await probe_reference(sample_id, out_root / "artifacts")
        if probe is None:
            continue

        core_scores = await evaluate_core_states(sample_id, probe["states"], out_root / "artifacts" / "preselect")
        if len(core_scores) < 2:
            continue

        if all(score >= 0.85 for score in core_scores):
            early_pass.append(probe)
        elif all(score < 0.85 for score in core_scores):
            early_fail.append(probe)

        if len(early_pass) >= limit // 2 and len(early_fail) >= limit // 2:
            break

    selected.extend(early_fail[: limit // 2])
    selected.extend(early_pass[: limit // 2])

    results = []
    for item in selected:
        results.append(await evaluate_sample(item["sample_id"], item["states"], out_root / "artifacts"))

    threshold_summaries = {
        "0.85": summarize_threshold(results, 0.85),
        "0.90": summarize_threshold(results, 0.90),
    }

    payload = {
        "selection": {
            "early_fail_count": len(early_fail[: limit // 2]),
            "early_pass_count": len(early_pass[: limit // 2]),
        },
        "selected_samples": selected,
        "results": results,
        "threshold_summaries": threshold_summaries,
    }
    write_json(out_root / "results.json", payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()
    asyncio.run(main_async(args.limit))


if __name__ == "__main__":
    main()
