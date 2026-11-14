"""How viral is a story? Three signals, each 0-1, combined, then freshness and topic multipliers.

- coverage:   distinct outlets that ran it, weighted by outlet weight (log scale, saturating)
- engagement: Hacker News points + 2 x comments, Reddit upvotes + 2 x comments (log scale, saturating)
- velocity:   engagement per hour since the story first appeared

They combine as 1 - (1 - wc*coverage)(1 - we*engagement)(1 - wv*velocity), so a Finnish story that five
outlets ran but nobody posted to Hacker News can still rank high, and a Hacker News hit that no outlet covered
yet can too, while a story strong on both ranks highest.

- freshness:  multiplies the result; halves every `half_life_hours`, never below 1 - recency_weight
- topic:      multiplies by the weight of the best-matching topic in config.yaml
"""
from __future__ import annotations

import datetime as dt
import math

from .models import Story
from .text import has_keyword


def _hours_since(iso: str, now: dt.datetime) -> float:
    try:
        t = dt.datetime.fromisoformat(iso)
    except (TypeError, ValueError):
        return 24.0
    return max(0.0, (now - t).total_seconds() / 3600)


def _sat(x: float, full: float) -> float:
    return min(1.0, math.log1p(max(0.0, x)) / math.log1p(full)) if full > 0 else 0.0


def match_topics(story: Story, cfg: dict) -> list[tuple[str, float]]:
    text = story.text()
    return [(t["name"], float(t.get("weight", 1.0))) for t in cfg.get("topics", [])
            if any(has_keyword(text, k) for k in t.get("any", []))]


def _n(x: int) -> str:
    return f"{x / 1000:.1f}k" if x >= 1000 else str(x)


def score(story: Story, cfg: dict, now: dt.datetime) -> None:
    sc = cfg.get("scoring", {})
    topics = match_topics(story, cfg)
    outlets = story.outlet_weights()
    if topics:
        best = max(topics, key=lambda t: t[1])
        story.topic, story.topic_weight = best
        story.topics = [t[0] for t in topics]
    elif len(outlets) >= cfg.get("untopical_min_outlets", 6):
        story.topic, story.topic_weight, story.topics = "Breaking (other)", 0.6, []
    else:
        story.topic, story.topic_weight, story.topics = "", 0.0, []

    eng = story.engagement
    age = _hours_since(story.first_published, now) if story.first_published else 24.0
    cov = _sat(sum(outlets.values()), sc.get("coverage_saturation", 8))
    en = _sat(eng, sc.get("engagement_saturation", 3000))
    vel = _sat(eng / max(1.0, age), sc.get("engagement_saturation", 3000) / 12)
    wc, we, wv = sc.get("coverage", 0.8), sc.get("engagement", 0.6), sc.get("velocity", 0.3)
    mix = 1 - (1 - wc * cov) * (1 - we * en) * (1 - wv * vel)
    rw = sc.get("recency_weight", 0.6)
    fresh = (1 - rw) + rw * 0.5 ** (age / sc.get("half_life_hours", 18))
    s = 100 * mix * fresh * story.topic_weight
    if story.status == "rising":
        s += sc.get("rising_bonus", 8)
    story.score = round(min(100.0, s), 1)
    story.signals = {"coverage": round(cov, 3), "engagement": round(en, 3), "velocity": round(vel, 3),
                     "freshness": round(fresh, 3), "age_hours": round(age, 1)}

    r = []
    names = sorted(outlets, key=lambda k: -outlets[k])
    if names:
        r.append(f"{len(names)} outlet{'s' if len(names) != 1 else ''}: " + ", ".join(names[:4])
                 + (f" +{len(names) - 4} more" if len(names) > 4 else ""))
    for src, label in (("hackernews", "HN"), ("reddit", "Reddit")):
        pts = sum(a.points for a in story.articles if a.source == src)
        com = sum(a.comments for a in story.articles if a.source == src)
        if pts:
            r.append(f"{label} {_n(pts)} {'points' if src == 'hackernews' else 'upvotes'}, {_n(com)} comments")
    r.append(f"first seen {age:.0f}h ago" if age >= 1 else "broke in the last hour")
    if story.topic:
        r.append(f"topic: {story.topic}")
    if story.status == "rising":
        r.append(f"rising: {len(outlets) - story.outlets_yesterday} more outlets since yesterday")
    story.reasons = r
