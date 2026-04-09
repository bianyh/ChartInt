import asyncio
import json
import os
import re
from copy import deepcopy
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import numpy as np
from openai import OpenAI
from PIL import Image
from playwright.async_api import async_playwright
from skimage.metrics import structural_similarity


DEFAULT_BASE_URL = os.environ.get("CHARTINT_BASE_URL", "https://kit.xin/v1")
DEFAULT_MODEL = os.environ.get("CHARTINT_MODEL", "gpt-5.4")


def require_api_key() -> str:
    api_key = os.environ.get("CHARTINT_API_KEY")
    if not api_key:
        raise RuntimeError("Missing CHARTINT_API_KEY in environment.")
    return api_key


def make_client() -> OpenAI:
    return OpenAI(
        api_key=require_api_key(),
        base_url=DEFAULT_BASE_URL,
        timeout=120,
    )


def stream_chat_completion(
    messages: list[dict[str, Any]],
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0,
) -> str:
    last_error = None
    for _ in range(3):
        try:
            client = make_client()
            stream = client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True,
                temperature=temperature,
            )
            parts: list[str] = []
            for chunk in stream:
                for choice in getattr(chunk, "choices", []) or []:
                    delta = getattr(choice, "delta", None)
                    if delta and getattr(delta, "content", None) is not None:
                        parts.append(delta.content)
            return "".join(parts)
        except Exception as error:  # noqa: BLE001
            last_error = error
    raise last_error


def extract_json_block(text: str) -> dict[str, Any]:
    content = text.strip()
    if not content:
        raise ValueError("Empty model output.")

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*(\{[\s\S]*\})\s*```", content)
    if fenced:
        return json.loads(fenced.group(1))

    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in model output.")
    return json.loads(content[start : end + 1])


def normalize_text(text: str) -> str:
    collapsed = re.sub(r"\s+", " ", text or "").strip().lower()
    return collapsed


def text_similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, normalize_text(left), normalize_text(right)).ratio()


def image_ssim(left_path: str, right_path: str, size: tuple[int, int] = (320, 240)) -> float:
    left = Image.open(left_path).convert("L").resize(size)
    right = Image.open(right_path).convert("L").resize(size)
    left_arr = np.asarray(left, dtype=np.float32) / 255.0
    right_arr = np.asarray(right, dtype=np.float32) / 255.0
    return float(structural_similarity(left_arr, right_arr, data_range=1.0))


def ensure_dir(path: str | Path) -> Path:
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def option_to_html(option: dict[str, Any]) -> str:
    option_json = json.dumps(option, ensure_ascii=False, indent=2)
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>ChartInt Multi-Step Edit</title>
  <script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>
  <style>
    html, body {{
      margin: 0;
      width: 100%;
      height: 100%;
      background: #ffffff;
      overflow: hidden;
      font-family: sans-serif;
    }}
    #chart {{
      width: 1200px;
      height: 800px;
      margin: 0 auto;
    }}
  </style>
</head>
<body>
  <div id="chart"></div>
  <script>
    const chart = echarts.init(document.getElementById('chart'));
    const option = {option_json};
    chart.setOption(option);
  </script>
</body>
</html>
"""


async def _open_chart(page, html_path: str) -> dict[str, Any]:
    await page.goto("file://" + os.path.abspath(html_path), wait_until="domcontentloaded")
    await page.wait_for_timeout(2200)
    return await page.evaluate(
        """() => {
          const div = document.querySelector('#main')
            || document.querySelector('#chart')
            || document.querySelector('div[id]');
          const inst = window.echarts && div ? echarts.getInstanceByDom(div) : null;
          if (!inst) {
            return {
              ok: false,
              divIds: [...document.querySelectorAll('div[id]')].map(node => node.id)
            };
          }

          const option = inst.getOption() || {};
          let dataLen = 0;
          let usableSeriesIndex = 0;
          (option.series || []).forEach((series, idx) => {
            const len = Array.isArray(series.data) ? series.data.length : 0;
            if (len > dataLen) {
              dataLen = len;
              usableSeriesIndex = idx;
            }
          });

          return {
            ok: true,
            dataLen,
            usableSeriesIndex,
            seriesCount: (option.series || []).length
          };
        }"""
    )


async def _capture_state(
    page,
    data_index: int,
    screenshot_path: str,
) -> dict[str, Any]:
    state_meta = await page.evaluate(
        """(payload) => {
          const div = document.querySelector('#main')
            || document.querySelector('#chart')
            || document.querySelector('div[id]');
          const inst = window.echarts && div ? echarts.getInstanceByDom(div) : null;
          if (!inst) {
            return { ok: false, error: 'no_instance' };
          }

          const option = inst.getOption() || {};
          const seriesList = option.series || [];
          let seriesIndex = 0;
          for (let idx = 0; idx < seriesList.length; idx += 1) {
            const series = seriesList[idx];
            if (Array.isArray(series.data) && series.data.length > payload.dataIndex) {
              seriesIndex = idx;
              break;
            }
          }

          try {
            inst.dispatchAction({ type: 'hideTip' });
          } catch (error) {}

          try {
            inst.dispatchAction({ type: 'downplay' });
          } catch (error) {}

          try {
            inst.dispatchAction({
              type: 'highlight',
              seriesIndex,
              dataIndex: payload.dataIndex
            });
          } catch (error) {}

          try {
            inst.dispatchAction({
              type: 'showTip',
              seriesIndex,
              dataIndex: payload.dataIndex
            });
          } catch (error) {}

          return { ok: true, seriesIndex };
        }""",
        {"dataIndex": data_index},
    )
    await page.wait_for_timeout(700)

    tooltip_payload = await page.evaluate(
        """() => {
          const nodes = [...document.querySelectorAll('div')]
            .map(node => {
              const style = getComputedStyle(node);
              return {
                text: (node.innerText || '').trim(),
                position: style.position,
                zIndex: style.zIndex,
                styleText: node.getAttribute('style') || ''
              };
            })
            .filter(item => item.text.length > 0)
            .sort((left, right) => right.text.length - left.text.length);
          return nodes.slice(0, 5);
        }"""
    )
    tooltip_text = tooltip_payload[0]["text"] if tooltip_payload else ""
    await page.screenshot(path=screenshot_path)
    return {
        "tooltip_text": tooltip_text,
        "tooltip_candidates": tooltip_payload,
        "series_index": state_meta.get("seriesIndex", 0),
        "ok": state_meta.get("ok", False),
    }


async def snapshot_chart_states(
    html_path: str,
    state_indices: list[int],
    output_dir: str | Path,
) -> dict[str, Any]:
    output_dir = ensure_dir(output_dir)
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            args=["--disable-web-security", "--allow-file-access-from-files"]
        )
        page = await browser.new_page(viewport={"width": 1200, "height": 800})
        meta = await _open_chart(page, html_path)
        if not meta.get("ok"):
            await browser.close()
            return {"ok": False, "meta": meta, "states": []}

        states: list[dict[str, Any]] = []
        for idx in state_indices:
            screenshot_path = str(output_dir / f"state_{idx}.png")
            captured = await _capture_state(page, idx, screenshot_path)
            states.append(
                {
                    "data_index": idx,
                    "tooltip_text": captured["tooltip_text"],
                    "tooltip_candidates": captured["tooltip_candidates"],
                    "series_index": captured["series_index"],
                    "screenshot_path": screenshot_path,
                    "ok": captured["ok"],
                }
            )
        await browser.close()
        return {"ok": True, "meta": meta, "states": states}


async def render_html_to_png(html_path: str, screenshot_path: str) -> dict[str, Any]:
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            args=["--disable-web-security", "--allow-file-access-from-files"]
        )
        page = await browser.new_page(viewport={"width": 1200, "height": 800})
        page_errors: list[str] = []
        page.on("pageerror", lambda exc: page_errors.append(str(exc)))
        await page.goto("file://" + os.path.abspath(html_path), wait_until="domcontentloaded")
        await page.wait_for_timeout(1800)
        has_instance = await page.evaluate(
            """() => {
              const div = document.querySelector('#main')
                || document.querySelector('#chart')
                || document.querySelector('div[id]');
              const inst = window.echarts && div ? echarts.getInstanceByDom(div) : null;
              return Boolean(inst);
            }"""
        )
        await page.screenshot(path=screenshot_path)
        await browser.close()
        return {
            "render_ok": bool(has_instance and not page_errors),
            "page_errors": page_errors,
            "screenshot_path": screenshot_path,
        }


async def extract_normalized_option_from_html(html_path: str) -> dict[str, Any]:
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            args=["--disable-web-security", "--allow-file-access-from-files"]
        )
        page = await browser.new_page(viewport={"width": 1200, "height": 800})
        await page.goto("file://" + os.path.abspath(html_path), wait_until="domcontentloaded")
        await page.wait_for_timeout(1800)
        payload = await page.evaluate(
            """() => {
              const div = document.querySelector('#main')
                || document.querySelector('#chart')
                || document.querySelector('div[id]');
              const inst = window.echarts && div ? echarts.getInstanceByDom(div) : null;
              if (!inst) {
                return { ok: false, option: null };
              }
              return { ok: true, option: inst.getOption() };
            }"""
        )
        await browser.close()
        return payload


async def render_and_extract_html(html_path: str, screenshot_path: str) -> dict[str, Any]:
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            args=["--disable-web-security", "--allow-file-access-from-files"]
        )
        page = await browser.new_page(viewport={"width": 1200, "height": 800})
        page_errors: list[str] = []
        page.on("pageerror", lambda exc: page_errors.append(str(exc)))
        await page.goto("file://" + os.path.abspath(html_path), wait_until="domcontentloaded")
        await page.wait_for_timeout(1800)
        payload = await page.evaluate(
            """() => {
              const div = document.querySelector('#main')
                || document.querySelector('#chart')
                || document.querySelector('div[id]');
              const inst = window.echarts && div ? echarts.getInstanceByDom(div) : null;
              if (!inst) {
                return { ok: false, option: null };
              }
              return { ok: true, option: inst.getOption() };
            }"""
        )
        await page.screenshot(path=screenshot_path)
        await browser.close()
        return {
            "render_ok": bool(payload.get("ok") and not page_errors),
            "page_errors": page_errors,
            "normalized_ok": bool(payload.get("ok")),
            "normalized_option": payload.get("option"),
            "screenshot_path": screenshot_path,
        }


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def deep_copy_option(option: dict[str, Any]) -> dict[str, Any]:
    return deepcopy(option)
