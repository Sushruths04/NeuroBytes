"""Decision log. FROZEN CONTRACT - sub-agents must not edit this file.

    from winfuse.common.decisions import log_decision

    log_decision(
        agent="A",
        title="Welch PSD segment length",
        choice="1.0 s Hann segments, 50% overlap (nperseg=250 at 250 Hz)",
        alternatives=[
            "0.5 s segments: finer time resolution, but 2 Hz bin width smears the delta band",
            "full 3 s window: 0.33 Hz bins, but only one segment so no variance averaging",
        ],
        rationale="1 Hz bins resolve the 0.5-4 Hz delta edge; 5 overlapping segments "
                  "in a 3 s epoch give usable variance reduction.",
        consequences="Delta power is the least reliable band because only ~3 bins fall "
                     "inside it. Flagged in the band-power summary.",
    )

Appends to `decisions/<agent>.md`. One call per material choice. Call it again
when you change your mind and pass `post_hoc=True` if the change was made after
seeing results -- Phase 2 renders those with a visible flag.
"""
from __future__ import annotations

import datetime as _dt
import json
import re
from pathlib import Path
from typing import Iterable, Sequence

from .config import DECISIONS_DIR

_SLUG = re.compile(r"[^a-zA-Z0-9_.-]+")


def _agent_file(agent: str) -> Path:
    slug = _SLUG.sub("_", str(agent).strip()) or "unknown"
    return DECISIONS_DIR / f"{slug}.md"


def _agent_jsonl(agent: str) -> Path:
    slug = _SLUG.sub("_", str(agent).strip()) or "unknown"
    return DECISIONS_DIR / f"{slug}.jsonl"


def _already_logged(agent: str, title: str) -> bool:
    """True if this (agent, title) pair is already in the agent's jsonl mirror."""
    p = _agent_jsonl(agent)
    if not p.exists():
        return False
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("agent") == agent and rec.get("title") == title:
            return True
    return False


def log_decision(
    agent: str,
    title: str,
    choice: str,
    alternatives: Sequence[str] | str | None = None,
    rationale: str = "",
    consequences: str = "",
    post_hoc: bool = False,
    tags: Iterable[str] = (),
    allow_duplicate: bool = False,
) -> Path:
    """Append one timestamped decision entry. Returns the file written to.

    Idempotent by default: if an entry with the same (agent, title) already exists,
    nothing is written. This makes the `log_*` scripts safe to re-run or import
    without duplicating the log. Pass `allow_duplicate=True` to record a genuine
    revision of an earlier decision (and set `post_hoc=True` if it followed a result).

    Parameters
    ----------
    agent : "orchestrator", "A", "B", "C"
    title : short noun phrase, e.g. "Blink detection threshold"
    choice : what was actually done, concretely, with numbers
    alternatives : the options rejected, each with why it lost
    rationale : why `choice` won
    consequences : what this costs, what it risks, what it makes impossible
    post_hoc : True if decided or revised AFTER seeing results
    """
    if isinstance(alternatives, str):
        alternatives = [alternatives]
    alts = list(alternatives or [])
    tags = list(tags)

    if not allow_duplicate and _already_logged(agent, title):
        return _agent_file(agent)

    ts = _dt.datetime.now().astimezone().isoformat(timespec="seconds")
    p = _agent_file(agent)
    new = not p.exists()

    lines: list[str] = []
    if new:
        lines += [
            f"# Decision log - agent `{agent}`",
            "",
            "Appended by `src.common.decisions.log_decision`. Newest entries at the bottom.",
            "Merged into the repo-root `decisions.md` in Phase 2.",
            "",
        ]
    flag = " **[post-hoc]**" if post_hoc else ""
    tagstr = f"  \n`tags: {', '.join(tags)}`" if tags else ""
    lines += [
        f"### [{ts}] [{agent}] {title}{flag}{tagstr}",
        "",
        f"**Decision** — {choice}",
        "",
        "**Alternatives considered**",
        "",
    ]
    if alts:
        lines += [f"- {a}" for a in alts]
    else:
        lines.append("- *(none recorded)*")
    lines += [
        "",
        f"**Why** — {rationale or '*(not recorded)*'}",
        "",
        f"**Consequences/risks** — {consequences or '*(not recorded)*'}",
        "",
    ]

    with open(p, "a") as f:
        f.write("\n".join(lines) + "\n")

    # machine-readable mirror, used by the Phase 2 merge to sort reliably
    jp = _agent_jsonl(agent)
    rec = {
        "timestamp": ts,
        "agent": agent,
        "title": title,
        "choice": choice,
        "alternatives": alts,
        "rationale": rationale,
        "consequences": consequences,
        "post_hoc": bool(post_hoc),
        "tags": tags,
    }
    with open(jp, "a") as f:
        f.write(json.dumps(rec) + "\n")
    return p


def load_all() -> list[dict]:
    """Every decision from every agent's .jsonl, sorted by timestamp."""
    recs: list[dict] = []
    for p in sorted(DECISIONS_DIR.glob("*.jsonl")):
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    recs.sort(key=lambda r: r.get("timestamp", ""))
    return recs


def render_entry(rec: dict) -> str:
    flag = " **[post-hoc]**" if rec.get("post_hoc") else ""
    L = [f"### [{rec.get('timestamp','?')}] [{rec.get('agent','?')}] {rec.get('title','?')}{flag}", ""]
    if rec.get("tags"):
        L += [f"`tags: {', '.join(rec['tags'])}`", ""]
    L += [f"**Decision** — {rec.get('choice','')}", "", "**Alternatives considered**", ""]
    alts = rec.get("alternatives") or []
    L += [f"- {a}" for a in alts] if alts else ["- *(none recorded)*"]
    L += [
        "",
        f"**Why** — {rec.get('rationale') or '*(not recorded)*'}",
        "",
        f"**Consequences/risks** — {rec.get('consequences') or '*(not recorded)*'}",
        "",
    ]
    return "\n".join(L)
