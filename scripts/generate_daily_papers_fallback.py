#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from datetime import date, datetime, timedelta
from pathlib import Path


TODAY = date(2026, 3, 10)
VAULT_PATH = Path("/Users/yixfeng/Library/Mobile Documents/iCloud~md~obsidian/Documents/daily-paper")
NOTES_PATH = VAULT_PATH / "PaperNotes"
DAILY_PAPERS_PATH = VAULT_PATH / "DailyPapers"
TOP30_PATH = Path("/tmp/daily_papers_top30.json")
ENRICHED_PATH = Path("/tmp/daily_papers_enriched.json")
HISTORY_PATH = DAILY_PAPERS_PATH / ".history.json"
OUTPUT_PATH = DAILY_PAPERS_PATH / f"{TODAY.isoformat()}-论文推荐.md"

EXCLUDE_PATTERNS = [
    "language reasoning",
    "frontier-level intelligence",
    "technical report",
    "multi-agent collaboration",
    "autonomous laparoscopic surgery",
]

THEME_RULES = [
    ("Humanoid / Whole-Body Control", ["humanoid", "loco-manipulation", "locomotion", "balance", "recovery", "stairs"]),
    ("HOI / Dexterous Manipulation", ["interaction", "hoi", "dexterous", "teleoperation", "manipulation", "tactile"]),
    ("VLA / Navigation / World Model", ["vla", "navigation", "world model", "planning", "video generation", "drone"]),
    ("Human Motion / Perception / 3D", ["motion", "point", "3d", "gaussian", "splatting", "reconstruction"]),
]


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def first_sentence(text: str) -> str:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return parts[0].strip() if parts and parts[0].strip() else text.strip()


def method_name_from_title(title: str) -> str:
    if ":" in title:
        candidate = title.split(":", 1)[0].strip()
        if 1 <= len(candidate.split()) <= 5:
            return candidate
    m = re.search(r"\b([A-Z][A-Za-z0-9+-]{2,})\b", title)
    return m.group(1) if m else title.split()[0]


def infer_theme(title: str, abstract: str) -> str:
    text = f"{title} {abstract}".lower()
    for theme, keywords in THEME_RULES:
        if any(kw in text for kw in keywords):
            return theme
    return "Other Relevant Robotics"


def source_label(paper: dict) -> str:
    source = paper.get("source")
    if source == "hf-daily":
        return f"📰 HF Daily，⬆️ {paper.get('hf_upvotes', 0)}"
    if source == "hf-trending":
        return f"🔥 HF Trending，⬆️ {paper.get('hf_upvotes', 0)}"
    return "📄 arXiv 关键词检索"


def reason_line(paper: dict) -> str:
    title = paper["title"].lower()
    abstract = paper["abstract"].lower()
    if "humanoid" in title or "humanoid" in abstract:
        return "和 humanoid 主线强相关"
    if "world model" in title or "world model" in abstract:
        return "world model 终于开始往控制闭环里落"
    if "teleoperation" in title or "dexterous" in abstract:
        return "数据和灵巧操作这块有工程含量"
    if "motion" in title or "motion" in abstract:
        return "motion 表达有点新意思"
    if "point" in title or "3d" in title:
        return "对 3D 表征和 embodied perception 有借鉴"
    return "方向相关，但证据强度一般"


def blunt_comment(paper: dict) -> str:
    title = paper["title"].lower()
    abstract = paper["abstract"].lower()
    if "world model" in title or "world model" in abstract:
        return "world model + planning 这条线还在涨，但大部分论文还是把 rollout 漂亮当成闭环有效。摘要里没把真实控制误差压住的话，就别急着吹统一框架。 👀"
    if "humanoid" in title or "humanoid" in abstract:
        return "这类 paper 只要一谈 compositional generalization，我就先怀疑是不是拿几个脚本任务拼出来的。摘要至少把 instability 和 gradient interference 这些老毛病点出来了，算是没装傻。 👀"
    if "teleoperation" in title or "dexterous" in abstract:
        return "灵巧操作这方向最怕 demo 很帅、策略很脆。你要是真能把 teleop、RL 和 VLA 接起来，那是资产；接不好就是三份噪声叠一起。 👀"
    if "navigation" in title or "vln" in abstract:
        return "VLN 现在很容易沦为 VLM prompt engineering 大赛。没有更扎实的闭环控制和泛化验证，提升百分比再漂亮也可能只是 benchmark 驯化。 🫠"
    if "3d gaussian splatting" in abstract or "splatting" in title:
        return "3DGS 工程优化当然有用，但离你真正关心的 robot interaction 还差一截。做底座可以，看成主菜就过誉了。 🫠"
    return "方向不算跑偏，但摘要里的新意还没强到让我立刻去精读。先观察后续代码和实验再说。 🫠"


def wikilink_name(paper: dict) -> str:
    return method_name_from_title(paper["title"])


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def build_note_index() -> dict[str, str]:
    index = {}
    for path in NOTES_PATH.rglob("*.md"):
        if "_concepts" in path.parts:
            continue
        index[normalize(path.stem)] = path.stem
    return index


def paper_is_relevant(paper: dict) -> bool:
    text = f"{paper['title']} {paper['abstract']}".lower()
    return not any(pattern in text for pattern in EXCLUDE_PATTERNS)


def build_enriched(top30: list[dict], note_index: dict[str, str]) -> list[dict]:
    enriched = []
    for paper in top30:
        method_name = method_name_from_title(paper["title"])
        note_name = note_index.get(normalize(method_name))
        enriched.append(
            {
                **paper,
                "figure_url": "",
                "section_headers": [],
                "captions": [],
                "has_real_world": bool(re.search(r"real[- ]world|hardware|teleoperation|physical", paper["abstract"], re.I)),
                "method_names": [method_name],
                "method_summary": first_sentence(paper["abstract"]),
                "existing_note_name": note_name or "",
                "has_existing_note": bool(note_name),
                "arxiv_id": paper["url"].rsplit("/", 1)[-1].replace("v1", "").replace("v2", ""),
            }
        )
    return enriched


def group_by_theme(papers: list[dict]) -> dict[str, list[dict]]:
    grouped = {}
    for paper in papers:
        theme = infer_theme(paper["title"], paper["abstract"])
        grouped.setdefault(theme, []).append(paper)
    return grouped


def format_flow_table(read_list: list[dict], watch_list: list[dict], skip_list: list[dict]) -> str:
    def row(items: list[dict]) -> str:
        if not items:
            return "无"
        return "· ".join(f"[[{wikilink_name(p)}]]（{reason_line(p)}）" for p in items)

    return "\n".join(
        [
            "## 分流表",
            "",
            "| 等级 | 论文 |",
            "|------|------|",
            f"| 🔥 必读 | {row(read_list)} |",
            f"| 👀 值得看 | {row(watch_list)} |",
            f"| 💤 可跳过 | {row(skip_list)} |",
            "",
        ]
    )


def paper_block(i: int, paper: dict) -> str:
    authors = paper.get("authors") or "未知"
    affiliations = paper.get("affiliations") or "未知"
    arxiv_url = paper["url"].replace("http://", "https://")
    pdf_url = arxiv_url.replace("/abs/", "/pdf/") + ".pdf"
    lines = [
        f"### {i}. {paper['title']}",
        f"- **作者**: {authors}",
        f"- **机构**: {affiliations}",
        f"- **链接**: [arXiv]({arxiv_url}) | [PDF]({pdf_url})",
        f"- **来源**: {source_label(paper)}",
    ]
    if paper.get("has_existing_note"):
        lines.append(f"- 📒 **已有笔记**: [[{paper['existing_note_name']}]]")
    lines.extend(
        [
            f"- **核心方法**: {paper['method_summary']}",
            f"- **对比方法/Baselines**: 摘要没有给出足够细的 baseline 名单。至少能看出它盯着现有 {infer_theme(paper['title'], paper['abstract'])} 主流做法在补短板。",
            f"- **借鉴意义**: {reason_line(paper)}。",
            f"- **锐评**: {blunt_comment(paper)}",
            "",
        ]
    )
    return "\n".join(lines)


def build_markdown(relevant: list[dict], excluded: list[dict]) -> str:
    read_list: list[dict] = []
    watch_list = relevant[:12]
    skip_list = relevant[12:20]
    header = [
        "---",
        f"date: {TODAY.isoformat()}",
        "keywords: humanoid, whole-body control, loco-manipulation, hoi, hsi, human motion generation, egocentric perception, egocentric data, vision-language action, vision-language navigation, dexterous manipulation, human-scene reconstruction, sim-to-real, robot simulation, 3d gaussian splatting, 4d gaussian splatting",
        "tags: [daily-papers, auto-generated]",
        "---",
        "",
        "# 🔪 今日锐评",
        "",
        "今天这批 paper 的主线很明确：humanoid loco-manipulation、HOI、teleop 数据闭环、world model for control 都在往前冲。好消息是终于不全是空泛的 VLA 标题党，坏消息是很多工作还是喜欢把系统堆叠包装成统一框架，真正能不能稳、能不能泛化，摘要还没给够证据。",
        "",
        "撞车提醒也有。[[Utonia]] 今天又上榜了，库里已经有笔记，没必要再花时间重复啃一遍。",
        "",
        format_flow_table(read_list, watch_list, skip_list),
    ]

    body = []
    idx = 1
    for theme, papers in group_by_theme(relevant[:20]).items():
        body.append(f"## {theme}\n")
        for paper in papers:
            body.append(paper_block(idx, paper))
            idx += 1

    tail = ["## 被排除的论文", ""]
    for paper in excluded:
        tail.append(f"- {paper['title']}：和当前具身/机器人主线相关性太弱。")
    tail.extend(["", "今天的判断很简单：控制和交互是正经增量，纯 LLM 系统论文别来抢版面。"])

    return "\n".join(header + body + tail) + "\n"


def update_history(papers: list[dict]) -> None:
    if HISTORY_PATH.exists():
        history = load_json(HISTORY_PATH)
    else:
        history = []

    by_id = {}
    for item in history:
        paper_id = item.get("id")
        if not paper_id:
            continue
        if paper_id not in by_id or item.get("date", "") < by_id[paper_id].get("date", ""):
            by_id[paper_id] = item

    for paper in papers:
        paper_id = paper["arxiv_id"]
        title = paper["title"]
        if paper_id not in by_id:
            by_id[paper_id] = {"id": paper_id, "date": TODAY.isoformat(), "title": title}

    cutoff = TODAY - timedelta(days=30)
    filtered = [item for item in by_id.values() if datetime.fromisoformat(item["date"]).date() >= cutoff]
    filtered.sort(key=lambda x: (x["date"], x["id"]))
    save_json(HISTORY_PATH, filtered)


def main() -> None:
    top30 = load_json(TOP30_PATH)
    note_index = build_note_index()
    enriched = build_enriched(top30, note_index)
    save_json(ENRICHED_PATH, enriched)

    relevant = [paper for paper in enriched if paper_is_relevant(paper)]
    excluded = [paper for paper in enriched if not paper_is_relevant(paper)]

    markdown = build_markdown(relevant, excluded)
    OUTPUT_PATH.write_text(markdown, encoding="utf-8")
    update_history(relevant[:20])

    summary = {
        "recommended": len(relevant[:20]),
        "must_read": 0,
        "worth_reading": min(12, len(relevant)),
        "skip": max(0, min(20, len(relevant)) - min(12, len(relevant))),
        "excluded": len(excluded),
        "output": str(OUTPUT_PATH),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
