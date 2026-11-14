"""The morning digest: a Markdown page that becomes a GitHub issue (GitHub emails it to you)."""
from __future__ import annotations

BADGE = {"new": "🆕 new", "rising": "📈 rising", "steady": ""}


def _story(i: int, s: dict) -> list[str]:
    b = s.get("brief") or {}
    head = b.get("headline") or s["headline"]
    badge = BADGE.get(s.get("status", ""), "")
    lines = [f"### {i}. [{head}]({s['url']})", "",
             f"**{s['score']:.0f}/100** · {s['topic']}" + (f" · {badge}" if badge else "")
             + (f" · 🇫🇮 {b['finland_link']} Finland link" if b.get("finland_link") in ("direct", "indirect") else "")]
    if b:
        lines += ["", b["summary"], "", f"*Why it matters:* {b['why_it_matters']}", f"*Angle:* {b['angle']}"]
        if b.get("confidence") == "low":
            lines.append("*Low confidence: rests on one outlet or a social post. Check before using.*")
    lines += ["", "_" + " · ".join(s["reasons"]) + "_", ""]
    links = []
    for a in s["articles"][:8]:
        links.append(f"[{a['outlet']}]({a['url']})")
        if a.get("discussion_url"):
            links.append(f"[discussion]({a['discussion_url']})")
    lines += ["Sources: " + " · ".join(links), ""]
    return lines


def digest(payload: dict, dashboard_url: str = "", top: int = 15) -> str:
    stories = payload["stories"]
    f = payload["funnel"]
    out = [f"# Viral news radar, {payload['date']}", ""]
    out.append(f"{len(stories)} stories worth a look out of {f.get('stories', 0):,} found in "
               f"{f.get('articles_in_window', 0):,} articles from the last {payload.get('lookback_hours', 48)} hours. "
               f"{payload['new_today']} are new since yesterday.")
    if dashboard_url:
        out += ["", f"**[Open the dashboard]({dashboard_url})** for filters and the full list."]
    finland = [s for s in stories if s.get("topic") == "Finland & Nordic tech" or (s.get("brief") or {}).get("finland_link") == "direct"]
    out += ["", "## Top stories", ""]
    for i, s in enumerate(stories[:top], 1):
        out += _story(i, s)
    extra = [s for s in finland if s not in stories[:top]][:5]
    if extra:
        out += ["## Also from Finland and the Nordics", ""]
        for i, s in enumerate(extra, 1):
            out += _story(i, s)
    bad = [s for s in payload["sources"] if not s["ok"] or "failed:" in (s.get("detail") or "")]
    if bad:
        out += ["## Source problems", ""] + [f"- **{s['name']}**: {s['error'] or s['detail']}" for s in bad] + [""]
    u = payload.get("llm", {})
    out.append("---")
    if u.get("enabled"):
        out.append(f"Briefs by {u['model']} from headlines and feed snippets only; each was checked against a "
                   f"verbatim quote ({u['unverified']} discarded). Cost today ${u['cost_usd']:.4f}.")
    else:
        out.append("Briefs are off (set ANTHROPIC_API_KEY to turn them on). Headlines are the outlets' own.")
    return "\n".join(out) + "\n"
