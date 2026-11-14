"""Collect -> keep recent -> drop noise -> cluster into stories -> match to yesterday -> score -> brief -> write."""
from __future__ import annotations

import datetime as dt
import json
import math
from pathlib import Path

from . import cluster as clu
from . import llm, scoring
from .models import Article, Story
from .sources import SourceResult, feeds, hackernews, reddit
from .text import canonical_url, has_keyword


def collect(cfg: dict, only: set[str] | None = None) -> list[SourceResult]:
    src = cfg.get("sources", {})

    def on(name: str) -> bool:
        return src.get(name, {}).get("enabled", True) and (only is None or name in only)

    results = []
    if on("feeds"):
        results.append(feeds.fetch(cfg.get("feeds", [])))
    if on("hackernews"):
        results.append(hackernews.fetch(cfg.get("lookback_hours", 48), src.get("hackernews", {}).get("min_points", 40)))
    if on("reddit"):
        r = src.get("reddit", {})
        results.append(reddit.fetch(r.get("subreddits", []), r.get("min_score", 150)))
    return results


def prefilter(articles: list[Article], cfg: dict, now: dt.datetime, funnel: dict) -> list[Article]:
    """Drop articles outside the time window, undated ones, and headlines matching exclude_title."""
    since = (now - dt.timedelta(hours=cfg.get("lookback_hours", 48))).isoformat()
    future = (now + dt.timedelta(hours=2)).isoformat()
    excl = [k.lower() for k in cfg.get("exclude_title", [])]
    out, seen = [], set()
    for a in articles:
        if not a.published:
            funnel["undated"] += 1
            continue
        if not (since <= a.published <= future):
            funnel["too_old"] += 1
            continue
        t = a.title.lower()
        if any((k in t) if not k[-1].isalnum() else has_keyword(t, k) for k in excl):
            funnel["excluded"] += 1
            continue
        key = (a.outlet, canonical_url(a.url))
        if key in seen:
            funnel["duplicate"] += 1
            continue
        seen.add(key)
        out.append(a)
    return out


def _cos(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    return len(sa & sb) / math.sqrt(len(sa) * len(sb)) if sa and sb else 0.0


def match_ledger(stories: list[Story], ledger: dict, today: dt.date, first_run: bool, threshold: float = 0.5) -> int:
    """Give each story a stable id across days and mark it new / rising / steady. Returns the number new today.

    A story matches an earlier one when they share an article link or most of their key words. Each ledger
    entry keeps the outlet count per day, so "rising" means more outlets today than on the last day it was seen.
    """
    day = today.isoformat()
    url_index = {u: sid for sid, e in ledger.items() for u in e.get("urls", [])}
    taken: set[str] = set()
    new = 0
    for s in stories:
        urls = {canonical_url(a.url) for a in s.articles}
        tset = clu.story_terms(s)
        sid = next((url_index[u] for u in urls if u in url_index and url_index[u] not in taken), None)
        if sid is None:
            best = max(((k, _cos(tset, e.get("terms", []))) for k, e in ledger.items() if k not in taken),
                       key=lambda x: x[1], default=(None, 0.0))
            sid = best[0] if best[1] >= threshold else None
        if sid is None:
            sid = s.id
            while sid in ledger or sid in taken:
                sid += "x"
            ledger[sid] = {"first_seen": day, "by_day": {}}
        e = ledger[sid]
        earlier = sorted(d for d in e.get("by_day", {}) if d < day)
        prev = e["by_day"][earlier[-1]] if earlier else 0
        n_out = len(s.outlets)
        if e["first_seen"] == day:
            s.status = "steady" if first_run else "new"
            new += 0 if first_run else 1
        else:
            s.status = "rising" if n_out > prev else "steady"
        s.outlets_yesterday = prev
        taken.add(sid)
        s.id, s.first_seen = sid, e["first_seen"]
        by_day = {d: n for d, n in e.get("by_day", {}).items() if d >= (today - dt.timedelta(days=7)).isoformat()}
        by_day[day] = max(n_out, by_day.get(day, 0))
        e.update(last_seen=day, by_day=by_day, terms=tset, urls=sorted(urls)[:40], headline=s.headline)
    return new


def prune_ledger(ledger: dict, today: dt.date, keep_days: int = 14) -> None:
    cutoff = (today - dt.timedelta(days=keep_days)).isoformat()
    for k in [k for k, e in ledger.items() if e.get("last_seen", e.get("first_seen", "")) < cutoff]:
        del ledger[k]


def process(articles: list[Article], cfg: dict, now: dt.datetime, ledger: dict, first_run: bool) -> tuple[list[Story], dict, int]:
    funnel = {"collected": len(articles), "undated": 0, "too_old": 0, "excluded": 0, "duplicate": 0}
    arts = prefilter(articles, cfg, now, funnel)
    c = cfg.get("clustering", {})
    stories = clu.cluster(arts, c.get("threshold", 0.18), c.get("min_shared_terms", 1), c.get("dek_weight", 0.5),
                          c.get("nameless_penalty", 0.1))
    funnel["articles_in_window"] = len(arts)
    funnel["stories"] = len(stories)
    # Score once to find the candidates, then match only those against earlier days (cheap, and keeps the
    # ledger to stories worth remembering), then score again with "rising" known.
    bonus = cfg.get("scoring", {}).get("rising_bonus", 8)
    min_score = cfg.get("min_score", 0)
    cands, off_topic, low = [], 0, 0
    for s in stories:
        scoring.score(s, cfg, now)
        if not s.topic_weight:
            off_topic += 1
        elif s.score < min_score - bonus:
            low += 1
        else:
            cands.append(s)
    match_ledger(cands, ledger, now.date(), first_run)
    kept = []
    for s in cands:
        scoring.score(s, cfg, now)
        if s.score < min_score:
            low += 1
        else:
            kept.append(s)
    kept.sort(key=lambda s: (-s.score, s.headline.lower()))
    kept = kept[: cfg.get("max_stories", 80)]
    funnel.update(off_topic=off_topic, low_score=low, shown=len(kept))
    new_shown = sum(1 for s in kept if s.status == "new")
    return kept, funnel, new_shown


def run(cfg: dict, out_path: Path, now: dt.datetime | None = None, only: set[str] | None = None,
        ledger_path: Path | None = None, reader: "llm.Reader | None" = None,
        llm_cache_path: Path | None = None, results: list[SourceResult] | None = None) -> dict:
    now = now or dt.datetime.now(dt.timezone.utc)
    ledger_path = ledger_path or out_path.parent.parent / "data" / "seen.json"
    results = results if results is not None else collect(cfg, only)
    if results and not any(r.ok and r.articles for r in results):
        raise RuntimeError("no source returned anything: " + "; ".join(f"{r.name}: {r.error or r.detail}" for r in results))
    ledger: dict = {}
    if ledger_path.exists():
        try:
            ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        except ValueError:
            ledger = {}
    first_run = not ledger
    stories, funnel, new_today = process([a for r in results for a in r.articles], cfg, now, ledger, first_run)
    prune_ledger(ledger, now.date())

    llm_info = {"enabled": False}
    lc = cfg.get("llm", {})
    if reader is None and lc.get("enabled", True):
        reader = llm.Reader(model=lc.get("model", llm.DEFAULT_MODEL), audience=lc.get("audience", ""),
                            workers=lc.get("workers", 4), cache_path=llm_cache_path or ledger_path.with_name("llm_cache.json"))
    if reader is not None and reader.enabled:
        reader.read_all(stories[: lc.get("top_n", 25)], now.date().isoformat())
        reader.save()
        llm_info = {"enabled": True, "model": reader.model, **reader.usage.as_dict()}

    shown_by_source: dict[str, int] = {}
    for s in stories:
        for name in {a.source if a.source != "feed" else "feeds" for a in s.articles}:
            shown_by_source[name] = shown_by_source.get(name, 0) + 1
    payload = {
        "generated_at": now.isoformat(timespec="seconds"),
        "date": now.date().isoformat(),
        "new_today": new_today,
        "first_run": first_run,
        "lookback_hours": cfg.get("lookback_hours", 48),
        "funnel": funnel,
        "llm": llm_info,
        "sources": [{"name": r.name, "ok": r.ok, "fetched": len(r.articles), "in_stories": shown_by_source.get(r.name, 0),
                     "detail": r.detail, "error": r.error} for r in results],
        "topics": [t["name"] for t in cfg.get("topics", [])] + ["Breaking (other)"],
        "stories": [s.to_public() for s in stories],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_path.write_text(json.dumps(ledger, ensure_ascii=False, indent=0, sort_keys=True), encoding="utf-8")
    return payload
