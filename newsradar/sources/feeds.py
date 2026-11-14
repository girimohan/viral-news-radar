"""Publisher RSS 2.0 and Atom feeds, read with the standard library only."""
from __future__ import annotations

import concurrent.futures as cf
import datetime as dt
import hashlib
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime

from ..http import FetchError, get
from ..models import Article
from ..text import strip_html
from . import SourceResult

ATOM = "{http://www.w3.org/2005/Atom}"
DC = "{http://purl.org/dc/elements/1.1/}"
CONTENT = "{http://purl.org/rss/1.0/modules/content/}"


def parse_date(s: str | None) -> str:
    """RFC 822 (RSS) or ISO 8601 (Atom) to ISO 8601 UTC. Empty string when unreadable."""
    s = (s or "").strip()
    if not s:
        return ""
    d = None
    try:
        d = parsedate_to_datetime(s)
    except (TypeError, ValueError, IndexError):
        try:
            d = dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
        except ValueError:
            return ""
    if d is None:
        return ""
    if d.tzinfo is None:
        d = d.replace(tzinfo=dt.timezone.utc)
    return d.astimezone(dt.timezone.utc).isoformat(timespec="seconds")


def _text(el, *names) -> str:
    for n in names:
        x = el.find(n)
        if x is not None and (x.text or "").strip():
            return x.text.strip()
    return ""


def parse_feed(xml: bytes | str, feed: dict) -> list[Article]:
    """Items of one feed as Articles. `feed` is the entry from config.yaml (name, url, weight, region, lang)."""
    root = ET.fromstring(xml)
    out: list[Article] = []
    items = root.findall("./channel/item") or root.findall(".//item")
    atom = False
    if not items:
        items = root.findall(f"{ATOM}entry")
        atom = True
    for it in items:
        if atom:
            title = strip_html(_text(it, f"{ATOM}title"))
            link = ""
            for l in it.findall(f"{ATOM}link"):
                if l.get("rel", "alternate") == "alternate" and l.get("href"):
                    link = l.get("href")
                    break
            published = parse_date(_text(it, f"{ATOM}published", f"{ATOM}updated"))
            summary = strip_html(_text(it, f"{ATOM}summary", f"{ATOM}content"))
            guid = _text(it, f"{ATOM}id") or link
        else:
            title = strip_html(_text(it, "title"))
            link = _text(it, "link")
            if not link:
                g = it.find("guid")
                if g is not None and (g.text or "").startswith("http"):
                    link = g.text.strip()
            published = parse_date(_text(it, "pubDate", f"{DC}date"))
            summary = strip_html(_text(it, "description", f"{CONTENT}encoded"))
            guid = _text(it, "guid") or link
        if not title or not link:
            continue
        out.append(Article(
            source="feed", source_id=f"{feed['name']}:{hashlib.sha1(guid.encode()).hexdigest()[:12]}",
            outlet=feed["name"], title=title, url=link, published=published, summary=summary[:1500],
            region=feed.get("region", ""), lang=feed.get("lang", "en"), weight=float(feed.get("weight", 1.0)),
        ))
    return out


def fetch_one(feed: dict) -> tuple[dict, list[Article], str]:
    try:
        body = get(feed["url"], raw=True)
        if body is None:
            return feed, [], "feed not found (404)"
        return feed, parse_feed(body, feed), ""
    except (FetchError, ET.ParseError) as e:
        return feed, [], f"{type(e).__name__}: {e}"[:160]


def fetch(feeds: list[dict], workers: int = 8) -> SourceResult:
    res = SourceResult("feeds")
    broken = []
    with cf.ThreadPoolExecutor(workers) as ex:
        for feed, arts, err in ex.map(fetch_one, feeds):
            res.articles.extend(arts)
            if err:
                broken.append(f"{feed['name']} ({err})")
    res.detail = f"{len(feeds) - len(broken)}/{len(feeds)} feeds read"
    if broken:
        res.detail += "; failed: " + "; ".join(broken)
    if feeds and len(broken) == len(feeds):
        res.ok, res.error = False, "every feed failed"
    return res
