from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Iterable

from .models import SentenceScore


def _sentence_block(item: SentenceScore) -> str:
    cls = "needs-work" if item.worth_optimizing else "stable"
    chips = [
        f"score {item.score}",
        f"loss {item.loss}",
        f"logic {item.logic_score}",
        f"slot {item.slot_relevance_score}",
        f"redundancy {item.redundancy_score}",
    ]
    tags = " ".join(f'<span class="chip">{escape(tag)}</span>' for tag in chips)
    slots = "、".join(item.matched_slots) if item.matched_slots else "无"
    terms = "、".join(item.matched_terms) if item.matched_terms else "无"
    details = ""
    if item.worth_optimizing:
        details = (
            "<details class=\"details\">"
            "<summary>展开诊断与优化建议</summary>"
            f"<div class=\"detail-item\"><strong>reasoning:</strong> {escape(item.reasoning)}</div>"
            f"<div class=\"detail-item\"><strong>improvement:</strong> {escape(item.improvement_suggestion)}</div>"
            "</details>"
        )
    return (
        f'<div class="sentence {cls}">'
        f'<div class="sid">S{item.sentence_id}</div>'
        f'<div class="content">{escape(item.sentence_text)}</div>'
        f'<div class="meta">{tags}</div>'
        f'<div class="meta-line"><strong>matched_slots:</strong> {escape(slots)}</div>'
        f'<div class="meta-line"><strong>matched_terms:</strong> {escape(terms)}</div>'
        f'{details}'
        "</div>"
    )


def _panel(sentence_scores: Iterable[SentenceScore]) -> str:
    return "\n".join(_sentence_block(item) for item in sentence_scores)


def render_comparison_html(
    context_1_sentence_scores: list[SentenceScore],
    context_2_sentence_scores: list[SentenceScore],
    output_path: str,
    context_more_to_optimize: str,
) -> str:
    summary = {
        "context_1": "context_1 需要优化的句子更多。",
        "context_2": "context_2 需要优化的句子更多。",
        "tie": "两个文本需要优化的句子数量相同。",
    }[context_more_to_optimize]
    left = _panel(context_1_sentence_scores)
    right = _panel(context_2_sentence_scores)
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>TextScore Optimization View</title>
  <style>
    :root {{
      --bg: #efe9de;
      --paper: #fffdf8;
      --ink: #1e1a16;
      --muted: #6f655b;
      --border: #d9cfc3;
      --needs-work: #ffe6c8;
      --stable: #eef2eb;
      --chip: #f6efe4;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "PingFang SC", "Noto Serif SC", serif;
      background: radial-gradient(circle at top, #f7f1e4, var(--bg));
      color: var(--ink);
    }}
    .wrap {{
      max-width: 1440px;
      margin: 0 auto;
      padding: 32px 20px 48px;
    }}
    h1 {{ margin: 0 0 8px; font-size: 32px; }}
    p {{ margin: 0 0 24px; color: var(--muted); }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 20px;
    }}
    .panel {{
      background: rgba(255,255,255,0.78);
      border: 1px solid var(--border);
      border-radius: 18px;
      padding: 18px;
      backdrop-filter: blur(12px);
    }}
    .sentence {{
      border-radius: 14px;
      padding: 12px;
      margin-bottom: 12px;
      border: 1px solid var(--border);
    }}
    .sentence.needs-work {{ background: var(--needs-work); }}
    .sentence.stable {{ background: var(--stable); }}
    .sid {{ font-size: 12px; color: var(--muted); margin-bottom: 6px; }}
    .content {{ line-height: 1.7; font-size: 15px; margin-bottom: 8px; }}
    .meta {{ display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 8px; }}
    .chip {{
      display: inline-block;
      padding: 3px 8px;
      border-radius: 999px;
      background: var(--chip);
      border: 1px solid var(--border);
      font-size: 12px;
      color: #4d4339;
    }}
    .meta-line {{ font-size: 12px; color: #4d4339; margin-bottom: 6px; }}
    .details {{ margin-top: 10px; }}
    .details summary {{ cursor: pointer; color: #7b4d13; }}
    .detail-item {{ margin-top: 8px; font-size: 13px; color: #4d4339; line-height: 1.6; }}
    @media (max-width: 960px) {{
      .grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>TextScore 优化视图</h1>
    <p>{escape(summary)} 橙色表示值得优化，绿色表示当前优先级较低。点击橙色句子的折叠项可查看 reasoning 与优化建议。</p>
    <div class="grid">
      <section class="panel">
        <h2>context_1</h2>
        {left}
      </section>
      <section class="panel">
        <h2>context_2</h2>
        {right}
      </section>
    </div>
  </div>
</body>
</html>"""
    path = Path(output_path)
    path.write_text(html, encoding="utf-8")
    return str(path)
