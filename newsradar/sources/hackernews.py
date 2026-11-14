"""Hacker News stories from the last day or two, with points and comments, via the public Algolia API."""
from __future__ import annotations

import datetime as dt
import time

from ..http import FetchError, get
from ..models import Article
from . import SourceResult

API = "https://hn.algolia.com/api/v1/search"


def parse(data: dict, min_points: int = 0) -> list[Article]:
    out = []
    for h in data.get("hits") or []:
        pts = int(h.get("points") or 0)
        if pts < min_points or not h.get("title"):
            continue
        oid = str(h.get("objectID") or h.get("story_id"))
        thread = f"https://news.ycombinator.com/item?id={oid}"
        ts = h.get("created_at_i")
        published = dt.datetime.fromtimestamp(ts, dt.timezone.utc).isoformat(timespec="seconds") if ts else ""
        out.append(Article(
            source="hackernews", source_id=oid, outlet="Hacker News", title=h["title"].strip(),
            url=h.get("url") or thread, published=published, summary=(h.get("story_text") or "")[:600],
            region="Social", points=pts, comments=int(h.get("num_comments") or 0), discussion_url=thread,
        ))
    return out


def fetch(lookback_hours: int = 48, min_points: int = 40, now: float | None = None) -> SourceResult:
    since = int((now or time.time()) - lookback_hours * 3600)
    url = f"{API}?tags=story&numericFilters=created_at_i>{since},points>={min_points}&hitsPerPage=300"
    try:
        data = get(url) or {}
    except FetchError as e:
        return SourceResult("hackernews", ok=False, error=str(e))
    arts = parse(data, min_points)
    return SourceResult("hackernews", arts, detail=f"{len(arts)} stories with {min_points}+ points")
