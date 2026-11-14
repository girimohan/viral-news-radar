"""Morning briefs: Claude reads the headlines and snippets of the top stories and writes a short brief for each.

Guards, as in remote-job-radar:

* Evidence must be quoted verbatim from the headlines or snippets it was given. If the quote is not there,
  the brief is thrown away and the dashboard shows the outlets' own headlines instead. This stops the model
  "summarising" a story from what it remembers rather than from today's reporting.
* The model sees only headlines and feed snippets, never full articles, and is told so; briefs say what the
  outlets report, not what is true.
* Briefs are cached by the story's set of articles, so reruns are free and a story is re-read only when new
  outlets join it. `top_n` caps the calls per run.
"""
from __future__ import annotations

import concurrent.futures as cf
import hashlib
import json
import os
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests

from .models import Story

API_URL = "https://api.anthropic.com/v1/messages"
API_VERSION = "2023-06-01"
DEFAULT_MODEL = "claude-haiku-5-5"
PROMPT_VERSION = "brief-v1"     # bump when the prompt or schema changes, so the cache is rebuilt
MAX_ARTICLES = 12

SYSTEM = """You help a newsletter editor decide which of today's widely shared news stories to cover.
You get the headlines and short feed snippets that several outlets published about ONE story. You do not see
the full articles. Use only what these headlines and snippets say; do not add facts from memory.

Audience: {audience}

Return:
- headline: a plain, neutral headline for the story, max 14 words, no clickbait.
- summary: what happened, in at most 2 sentences, attributed where the snippets disagree ("X reports...").
- why_it_matters: one sentence on why readers would care.
- angle: one sentence suggesting how the editor could cover it for the audience above.
- finland_link: "direct" if the story is about Finland or a Finnish/Nordic organisation, "indirect" if it clearly
  affects Finland or the EU (e.g. EU regulation), "none" otherwise.
- category: one of models, products, policy, business, research, finland, security, other.
- evidence: copy exactly, character for character, the shortest headline or snippet fragment that supports
  the main fact in your summary.
- confidence: "high" if several outlets agree on the facts, "low" if it rests on one outlet, a rumour or a
  social post."""

SCHEMA = {
    "type": "object",
    "properties": {
        "headline": {"type": "string"},
        "summary": {"type": "string"},
        "why_it_matters": {"type": "string"},
        "angle": {"type": "string"},
        "finland_link": {"type": "string", "enum": ["direct", "indirect", "none"]},
        "category": {"type": "string", "enum": ["models", "products", "policy", "business", "research", "finland",
                                                 "security", "other"]},
        "evidence": {"type": "string"},
        "confidence": {"type": "string", "enum": ["high", "low"]},
    },
    "required": ["headline", "summary", "why_it_matters", "angle", "finland_link", "category", "evidence",
                 "confidence"],
    "additionalProperties": False,
}


class LLMError(Exception):
    pass


@dataclass
class Usage:
    calls: int = 0
    cached: int = 0
    failed: int = 0
    unverified: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def cost_usd(self, in_price: float = 0.10, out_price: float = 0.50) -> float:
        """Haiku 5.5 list prices per million tokens, as used in remote-job-radar."""
        return self.input_tokens / 1e6 * in_price + self.output_tokens / 1e6 * out_price

    def as_dict(self) -> dict:
        return {"calls": self.calls, "cached": self.cached, "failed": self.failed, "unverified": self.unverified,
                "input_tokens": self.input_tokens, "output_tokens": self.output_tokens,
                "cost_usd": round(self.cost_usd(), 6)}


def call_claude(system: str, user: str, schema: dict, *, model: str, api_key: str, max_tokens: int = 700,
                tries: int = 4, timeout: float = 60) -> tuple[dict, dict]:
    """One Messages API call with structured JSON output. Returns (parsed JSON, usage)."""
    body = {
        "model": model, "max_tokens": max_tokens, "system": system,
        "messages": [{"role": "user", "content": user}],
        "output_config": {"format": {"type": "json_schema", "schema": schema}},
    }
    headers = {"x-api-key": api_key, "anthropic-version": API_VERSION, "content-type": "application/json"}
    last = ""
    for attempt in range(tries):
        try:
            r = requests.post(API_URL, json=body, headers=headers, timeout=timeout)
        except requests.RequestException as e:
            last = type(e).__name__
            time.sleep(2 * (attempt + 1))
            continue
        if r.status_code in (429, 500, 502, 503, 529):
            last = f"HTTP {r.status_code}"
            time.sleep(float(r.headers.get("retry-after") or 5 * (attempt + 1)))
            continue
        if not r.ok:
            raise LLMError(f"HTTP {r.status_code}: {r.text[:200]}")
        data = r.json()
        if data.get("stop_reason") in ("refusal", "max_tokens"):
            raise LLMError(f"stop_reason {data['stop_reason']}")
        text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
        try:
            return json.loads(text), data.get("usage", {})
        except ValueError as e:
            raise LLMError("response was not JSON") from e
    raise LLMError(last or "no response")


_SPACE = re.compile(r"\s+")
_QUOTES = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-", " ": " "})


def _norm(s: str) -> str:
    return _SPACE.sub(" ", (s or "").translate(_QUOTES)).strip().lower()


def evidence_found(evidence: str, *texts: str) -> bool:
    """True when the quote appears in the material (ignoring case, whitespace and curly quotes).
    A quote of 12+ words also passes when its middle 80% is present (models trim the ends)."""
    ev = _norm(evidence).strip(" .\"'")
    if not ev:
        return False
    hay = _norm(" ".join(texts))
    if ev in hay:
        return True
    words = ev.split()
    if len(words) >= 12:
        cut = max(1, len(words) // 10)
        return " ".join(words[cut:-cut]) in hay
    return False


def material(story: Story) -> str:
    lines = []
    arts = sorted(story.articles, key=lambda a: (a.is_social, -a.weight))[:MAX_ARTICLES]
    for a in arts:
        extra = f" [{a.points} points, {a.comments} comments]" if a.is_social else ""
        lines.append(f"- {a.outlet}{extra} ({a.published[:16] or 'undated'}): {a.title}\n  {a.summary[:500]}")
    return "Headlines and snippets about one story:\n" + "\n".join(lines)


def cache_key(story: Story, model: str) -> str:
    raw = f"{PROMPT_VERSION}|{model}|" + "|".join(sorted(a.uid for a in story.articles))
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20]


class Reader:
    def __init__(self, model: str = DEFAULT_MODEL, audience: str = "", workers: int = 4,
                 cache_path: Path | None = None, api_key: str | None = None, call=call_claude):
        self.model, self.workers, self.cache_path, self.call = model, workers, cache_path, call
        self.api_key = api_key if api_key is not None else os.environ.get("ANTHROPIC_API_KEY", "")
        self.system = SYSTEM.format(audience=audience or "general readers interested in technology")
        self.usage = Usage()
        self.cache: dict = {}
        if cache_path and cache_path.exists():
            try:
                self.cache = json.loads(cache_path.read_text(encoding="utf-8"))
            except ValueError:
                self.cache = {}

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def read(self, story: Story, today: str) -> tuple[dict, str]:
        """Returns (brief, status) with status "checked", "unverified" or "failed"."""
        key = cache_key(story, self.model)
        hit = self.cache.get(key)
        if hit:
            with self.usage.lock:
                self.usage.cached += 1
            hit["used"] = today
            return hit["brief"], hit["status"]
        try:
            out, usage = self.call(self.system, material(story), SCHEMA, model=self.model, api_key=self.api_key)
        except LLMError:
            with self.usage.lock:
                self.usage.failed += 1
            return {}, "failed"
        texts = [t for a in story.articles for t in (a.title, a.summary)]
        status = "checked" if evidence_found(out.get("evidence", ""), *texts) else "unverified"
        with self.usage.lock:
            self.usage.calls += 1
            self.usage.input_tokens += int(usage.get("input_tokens", 0))
            self.usage.output_tokens += int(usage.get("output_tokens", 0))
            if status == "unverified":
                self.usage.unverified += 1
        brief = out if status == "checked" else {}
        self.cache[key] = {"brief": brief, "status": status, "used": today}
        return brief, status

    def read_all(self, stories: list[Story], today: str) -> None:
        with cf.ThreadPoolExecutor(max(1, self.workers)) as ex:
            for s, (brief, status) in zip(stories, ex.map(lambda s: self.read(s, today), stories)):
                s.brief, s.llm_status = brief, status

    def save(self, keep_days: int = 14) -> None:
        if not self.cache_path:
            return
        import datetime as dt
        cutoff = (dt.date.today() - dt.timedelta(days=keep_days)).isoformat()
        self.cache = {k: v for k, v in self.cache.items() if v.get("used", "") >= cutoff}
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self.cache, ensure_ascii=False, indent=0), encoding="utf-8")
