import argparse
import asyncio
import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

from extra_experiment_utils import (
    ensure_dir,
    extract_json_block,
    option_to_html,
    render_and_extract_html,
    stream_chat_completion,
    write_json,
)


ROOT = Path("/mnt/data_disk/bianyuhan/chartint_re/chart2code")
OUT_ROOT = ROOT / "extra_experiments" / "multistep_edit"


STEP_INSTRUCTIONS = [
    "Add a centered chart title with the exact text 'Quarterly Review Dashboard'. Keep all data values, series names, and chart types unchanged.",
    "Move the legend to the bottom center and keep it horizontal. Preserve every earlier edit.",
    "Change the chart background color to '#f7f7fb'. Preserve all earlier edits and all data.",
    "Turn on value labels for every series. Put labels on top for bar series and above points for line series. Preserve everything else.",
    "Replace the chart palette with ['#2456A6', '#2F8F6B', '#D9822B'] in this order. Preserve everything else.",
    "Enable dashed light-gray horizontal split lines on the y-axis using color '#d0d7e2'. Preserve everything else.",
    "Increase the plot padding so the grid becomes left=70, right=50, top=80, bottom=70. Preserve everything else.",
    "Make the series styles more prominent. For every bar series, set rounded top corners [6,6,0,0]. For every line series, set line width to 3 and symbol size to 10. Preserve everything else.",
    "Move the title to left=30 and top=10, and recolor the title text to '#16324f'. Keep the same title text and preserve everything else.",
    "Move the legend to the right side with right=12, top='middle', orient='vertical'. Preserve every earlier edit unless this instruction overrides it.",
]


ACTIVE_FIELDS_BY_STEP = {
    1: ["title_text"],
    2: ["title_text", "legend_pos"],
    3: ["title_text", "legend_pos", "background"],
    4: ["title_text", "legend_pos", "background", "labels_signature"],
    5: ["title_text", "legend_pos", "background", "labels_signature", "palette"],
    6: ["title_text", "legend_pos", "background", "labels_signature", "palette", "splitline"],
    7: ["title_text", "legend_pos", "background", "labels_signature", "palette", "splitline", "grid"],
    8: ["title_text", "legend_pos", "background", "labels_signature", "palette", "splitline", "grid", "series_style"],
    9: ["title_text", "legend_pos", "background", "labels_signature", "palette", "splitline", "grid", "series_style", "title_layout"],
    10: ["title_text", "legend_pos", "background", "labels_signature", "palette", "splitline", "grid", "series_style", "title_layout"],
}


def strip_code_fences(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:html|json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def first(value: Any) -> Any:
    if isinstance(value, list):
        return value[0] if value else {}
    return value or {}


def deep_get(mapping: dict[str, Any], path: list[str], default: Any = None) -> Any:
    current = mapping
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def make_base_option(chart_id: int) -> dict[str, Any]:
    categories = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]

    variants = [
        {
            "series": [
                {"name": "Revenue", "type": "bar", "data": [120, 132, 101, 134, 90, 230]},
                {"name": "Target", "type": "bar", "data": [110, 120, 115, 140, 100, 210]},
            ]
        },
        {
            "series": [
                {"name": "North", "type": "line", "data": [32, 45, 41, 53, 49, 60]},
                {"name": "South", "type": "line", "data": [28, 35, 39, 47, 43, 55]},
            ]
        },
        {
            "series": [
                {"name": "Orders", "type": "bar", "data": [22, 18, 29, 31, 24, 36]},
                {"name": "Conversion", "type": "line", "data": [12, 14, 15, 18, 17, 22]},
            ]
        },
        {
            "series": [
                {"name": "Desktop", "type": "bar", "data": [52, 61, 58, 63, 70, 74]},
                {"name": "Mobile", "type": "bar", "data": [45, 49, 51, 57, 64, 68]},
                {"name": "Tablet", "type": "bar", "data": [12, 14, 13, 15, 17, 18]},
            ]
        },
        {
            "series": [
                {"name": "Visits", "type": "line", "data": [220, 210, 260, 280, 310, 305]},
                {"name": "Qualified", "type": "line", "data": [90, 94, 110, 128, 140, 145]},
                {"name": "Won", "type": "line", "data": [32, 34, 39, 43, 47, 52]},
            ]
        },
    ]
    variant = deepcopy(variants[chart_id % len(variants)])
    option = {
        "title": {"text": f"Base Chart {chart_id + 1}", "left": "center", "top": 8},
        "tooltip": {"trigger": "axis"},
        "legend": {"top": 8, "left": "center"},
        "grid": {"left": 50, "right": 30, "top": 55, "bottom": 40},
        "xAxis": {
            "type": "category",
            "data": categories,
            "axisLabel": {"rotate": 0, "fontSize": 11, "color": "#475569"},
        },
        "yAxis": {
            "type": "value",
            "axisLabel": {"color": "#475569"},
            "splitLine": {"show": False},
        },
        "series": [],
    }

    for series in variant["series"]:
        if series["type"] == "bar":
            series["label"] = {"show": False, "position": "top"}
            series["itemStyle"] = {"borderRadius": [0, 0, 0, 0]}
            series["barWidth"] = 22
        if series["type"] == "line":
            series["label"] = {"show": False, "position": "top"}
            series["lineStyle"] = {"width": 2}
            series["symbol"] = "circle"
            series["symbolSize"] = 6
        option["series"].append(series)

    return option


def apply_reference_step(option: dict[str, Any], step_idx: int) -> dict[str, Any]:
    updated = deepcopy(option)

    if step_idx == 1:
        updated["title"]["text"] = "Quarterly Review Dashboard"
        updated["title"]["left"] = "center"
        updated["title"]["top"] = 10
    elif step_idx == 2:
        updated["legend"]["bottom"] = 10
        updated["legend"]["left"] = "center"
        updated["legend"]["orient"] = "horizontal"
        updated["legend"].pop("right", None)
        updated["legend"].pop("top", None)
    elif step_idx == 3:
        updated["backgroundColor"] = "#f7f7fb"
    elif step_idx == 4:
        for series in updated["series"]:
            series.setdefault("label", {})
            series["label"]["show"] = True
            series["label"]["position"] = "top"
    elif step_idx == 5:
        updated["color"] = ["#2456A6", "#2F8F6B", "#D9822B"]
    elif step_idx == 6:
        updated["yAxis"].setdefault("splitLine", {})
        updated["yAxis"]["splitLine"]["show"] = True
        updated["yAxis"]["splitLine"]["lineStyle"] = {
            "type": "dashed",
            "color": "#d0d7e2",
        }
    elif step_idx == 7:
        updated["grid"].update({"left": 70, "right": 50, "top": 80, "bottom": 70})
    elif step_idx == 8:
        for series in updated["series"]:
            if series["type"] == "bar":
                series.setdefault("itemStyle", {})
                series["itemStyle"]["borderRadius"] = [6, 6, 0, 0]
            if series["type"] == "line":
                series.setdefault("lineStyle", {})
                series["lineStyle"]["width"] = 3
                series["symbolSize"] = 10
    elif step_idx == 9:
        updated["title"]["left"] = 30
        updated["title"]["top"] = 10
        updated["title"].setdefault("textStyle", {})
        updated["title"]["textStyle"]["color"] = "#16324f"
    elif step_idx == 10:
        updated["legend"].pop("left", None)
        updated["legend"].pop("bottom", None)
        updated["legend"]["right"] = 12
        updated["legend"]["top"] = "middle"
        updated["legend"]["orient"] = "vertical"

    return updated


def semantic_signature(option: dict[str, Any]) -> dict[str, Any]:
    title = first(option.get("title"))
    legend = first(option.get("legend"))
    x_axis = first(option.get("xAxis"))
    y_axis = first(option.get("yAxis"))
    grid = first(option.get("grid"))
    series_list = option.get("series", []) or []

    labels_signature = []
    bar_styles = []
    line_styles = []
    for series in series_list:
        label = first(series.get("label"))
        labels_signature.append((series.get("type"), bool(label.get("show")), label.get("position")))

        if series.get("type") == "bar":
            item_style = first(series.get("itemStyle"))
            bar_styles.append(
                (
                    series.get("name"),
                    item_style.get("borderRadius"),
                    series.get("barWidth"),
                )
            )
        if series.get("type") == "line":
            line_style = first(series.get("lineStyle"))
            line_styles.append(
                (
                    series.get("name"),
                    line_style.get("width"),
                    series.get("symbolSize"),
                )
            )

    split_line = first(y_axis.get("splitLine"))
    split_line_style = first(split_line.get("lineStyle"))
    title_style = first(title.get("textStyle"))

    return {
        "title_text": title.get("text"),
        "title_layout": (title.get("left"), title.get("top"), title_style.get("color")),
        "legend_pos": {
            "left": legend.get("left"),
            "bottom_set": legend.get("bottom") is not None,
            "right_set": legend.get("right") is not None,
            "top": legend.get("top"),
            "orient": legend.get("orient"),
        },
        "background": option.get("backgroundColor"),
        "labels_signature": labels_signature,
        "palette": option.get("color"),
        "splitline": (
            bool(split_line.get("show")),
            split_line_style.get("type"),
            split_line_style.get("color"),
        ),
        "grid": (grid.get("left"), grid.get("right"), grid.get("top"), grid.get("bottom")),
        "series_style": {
            "bar": bar_styles,
            "line": line_styles,
        },
    }


def build_edit_prompt(current_option: dict[str, Any], instruction: str) -> list[dict[str, str]]:
    system = (
        "You are an expert Apache ECharts engineer editing a JSON option object. "
        "Return one strict JSON object only. "
        "Do not use markdown fences. "
        "Do not add comments. "
        "Keep all existing data values, series names, and chart types unchanged. "
        "Preserve all previous edits unless the new instruction explicitly overrides them."
    )
    option_json = json.dumps(current_option, ensure_ascii=False, indent=2)
    user = f"""Current ECharts option JSON:
{option_json}

Edit instruction:
{instruction}

Return only the updated complete JSON object."""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


async def run_chart_chain(chart_id: int, out_root: Path) -> dict[str, Any]:
    chart_dir = ensure_dir(out_root / f"chart_{chart_id:02d}")
    reference_dir = ensure_dir(chart_dir / "reference")
    generated_dir = ensure_dir(chart_dir / "generated")

    base_option = make_base_option(chart_id)
    current_generated_option = deepcopy(base_option)
    current_reference_option = base_option

    hard_collapse_step = None
    steps: list[dict[str, Any]] = []

    for step_idx, instruction in enumerate(STEP_INSTRUCTIONS, start=1):
        current_reference_option = apply_reference_step(current_reference_option, step_idx)
        reference_html = option_to_html(current_reference_option)
        reference_html_path = reference_dir / f"step_{step_idx:02d}.html"
        reference_html_path.write_text(reference_html, encoding="utf-8")
        reference_signature = semantic_signature(current_reference_option)

        if hard_collapse_step is not None:
            steps.append(
                {
                    "step": step_idx,
                    "instruction": instruction,
                    "hard_collapse": True,
                    "render_ok": False,
                    "field_fidelity": 0.0,
                    "exact_match": False,
                    "active_fields": ACTIVE_FIELDS_BY_STEP[step_idx],
                }
            )
            continue

        raw_json = stream_chat_completion(build_edit_prompt(current_generated_option, instruction), temperature=0.0)
        generated_json_path = generated_dir / f"step_{step_idx:02d}.json"
        generated_html_path = generated_dir / f"step_{step_idx:02d}.html"
        generated_png_path = generated_dir / f"step_{step_idx:02d}.png"
        generated_json_path.write_text(raw_json, encoding="utf-8")

        try:
            generated_option = extract_json_block(raw_json)
        except Exception:  # noqa: BLE001
            hard_collapse_step = step_idx
            steps.append(
                {
                    "step": step_idx,
                    "instruction": instruction,
                    "hard_collapse": True,
                    "render_ok": False,
                    "page_errors": ["invalid_json_output"],
                    "field_fidelity": 0.0,
                    "exact_match": False,
                    "active_fields": ACTIVE_FIELDS_BY_STEP[step_idx],
                }
            )
            continue

        generated_html = option_to_html(generated_option)
        generated_html_path.write_text(generated_html, encoding="utf-8")

        generated_payload = await render_and_extract_html(str(generated_html_path), str(generated_png_path))

        if not generated_payload["render_ok"] or not generated_payload["normalized_ok"]:
            hard_collapse_step = step_idx
            steps.append(
                {
                    "step": step_idx,
                    "instruction": instruction,
                    "hard_collapse": True,
                    "render_ok": False,
                    "page_errors": generated_payload["page_errors"],
                    "field_fidelity": 0.0,
                    "exact_match": False,
                    "active_fields": ACTIVE_FIELDS_BY_STEP[step_idx],
                }
            )
            continue

        current_generated_option = generated_option

        generated_signature = semantic_signature(generated_payload["normalized_option"])
        active_fields = ACTIVE_FIELDS_BY_STEP[step_idx]
        matched = sum(1 for field in active_fields if reference_signature[field] == generated_signature[field])
        fidelity = matched / len(active_fields)

        steps.append(
            {
                "step": step_idx,
                "instruction": instruction,
                "hard_collapse": False,
                "render_ok": True,
                "field_fidelity": round(fidelity, 4),
                "exact_match": fidelity == 1.0,
                "active_fields": active_fields,
                "reference_signature": {field: reference_signature[field] for field in active_fields},
                "generated_signature": {field: generated_signature[field] for field in active_fields},
            }
        )

    return {
        "chart_id": chart_id,
        "hard_collapse_step": hard_collapse_step,
        "steps": steps,
    }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    summary_rows = {}
    total = len(results)
    for step in [1, 3, 5, 8, 10]:
        step_records = [item["steps"][step - 1] for item in results]
        render_survival = sum(1 for row in step_records if row["render_ok"]) / total
        exact_match = sum(1 for row in step_records if row["exact_match"]) / total
        avg_fidelity = sum(row["field_fidelity"] for row in step_records) / total
        workflow_survival = sum(
            1
            for item in results
            if all(step_row["render_ok"] and step_row["exact_match"] for step_row in item["steps"][:step])
        ) / total
        summary_rows[step] = {
            "render_survival": round(render_survival, 4),
            "exact_match_rate": round(exact_match, 4),
            "average_field_fidelity": round(avg_fidelity, 4),
            "workflow_survival": round(workflow_survival, 4),
        }

    collapse_distribution = {}
    drift_distribution = {}
    for record in results:
        key = record["hard_collapse_step"] if record["hard_collapse_step"] is not None else "no_hard_crash"
        collapse_distribution[key] = collapse_distribution.get(key, 0) + 1
        drift_step = "no_drift"
        for step_row in record["steps"]:
            if not step_row["render_ok"] or not step_row["exact_match"]:
                drift_step = step_row["step"]
                break
        drift_distribution[drift_step] = drift_distribution.get(drift_step, 0) + 1

    return {
        "per_step": summary_rows,
        "collapse_distribution": collapse_distribution,
        "drift_distribution": drift_distribution,
    }


async def main_async(limit: int) -> None:
    out_root = ensure_dir(OUT_ROOT)
    results = []
    for chart_id in range(limit):
        results.append(await run_chart_chain(chart_id, out_root))

    payload = {
        "results": results,
        "summary": summarize(results),
    }
    write_json(out_root / "results.json", payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()
    asyncio.run(main_async(args.limit))


if __name__ == "__main__":
    main()
