"""Top Reddit posts of the day in chosen subreddits, through Reddit's official API.

Reddit only allows API use with registered credentials. Create a free "script" app at
https://www.reddit.com/prefs/apps and set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET. Without them
this source is skipped (reported on the dashboard, not an error).
"""
from __future__ import annotations

import datetime as dt
import os

import requests

from ..http import USER_AGENT, FetchError, get
from ..models import Article
from . import SourceResult

TOKEN_URL = "https://www.reddit.com/api/v1/access_token"


def token(client_id: str, secret: str) -> str:
    r = requests.post(TOKEN_URL, auth=(client_id, secret), data={"grant_type": "client_credentials"},
                      headers={"User-Agent": USER_AGENT}, timeout=20)
    if not r.ok:
        raise FetchError(f"Reddit token: HTTP {r.status_code}")
    return r.json()["access_token"]


def parse(data: dict, sub: str, min_score: int = 0) -> list[Article]:
    out = []
    for c in (data.get("data") or {}).get("children") or []:
        p = c.get("data") or {}
        if p.get("stickied") or p.get("over_18") or int(p.get("score") or 0) < min_score:
            continue
        thread = "https://www.reddit.com" + (p.get("permalink") or "")
        link = p.get("url_overridden_by_dest") or p.get("url") or thread
        ts = p.get("created_utc")
        out.append(Article(
            source="reddit", source_id=str(p.get("id")), outlet=f"r/{sub}", title=(p.get("title") or "").strip(),
            url=link, published=dt.datetime.fromtimestamp(ts, dt.timezone.utc).isoformat(timespec="seconds") if ts else "",
            summary=(p.get("selftext") or "")[:600], region="Social",
            points=int(p.get("score") or 0), comments=int(p.get("num_comments") or 0), discussion_url=thread,
        ))
    return out


def fetch(subreddits: list[str], min_score: int = 150) -> SourceResult:
    cid, secret = os.environ.get("REDDIT_CLIENT_ID", ""), os.environ.get("REDDIT_CLIENT_SECRET", "")
    if not (cid and secret):
        return SourceResult("reddit", detail="skipped: REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET not set")
    try:
        auth = {"Authorization": f"bearer {token(cid, secret)}"}
    except (FetchError, requests.RequestException, KeyError, ValueError) as e:
        return SourceResult("reddit", ok=False, error=str(e)[:160])
    res = SourceResult("reddit")
    failed = []
    for sub in subreddits:
        try:
            data = get(f"https://oauth.reddit.com/r/{sub}/top?t=day&limit=50", headers=auth) or {}
            res.articles.extend(parse(data, sub, min_score))
        except FetchError as e:
            failed.append(f"r/{sub} ({e})")
    res.detail = f"{len(subreddits) - len(failed)}/{len(subreddits)} subreddits read"
    if failed:
        res.detail += "; failed: " + "; ".join(failed)
    if subreddits and len(failed) == len(subreddits):
        res.ok, res.error = False, "every subreddit failed"
    return res
