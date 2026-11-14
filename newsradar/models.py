from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class Article:
    """One item from one source: a publisher's feed entry, a Hacker News post or a Reddit post."""
    source: str                 # "feed", "hackernews", "reddit"
    source_id: str
    outlet: str                 # "The Verge", "Hacker News", "r/artificial"
    title: str
    url: str                    # the article itself (for HN/Reddit: the link they point to)
    published: str = ""         # ISO 8601 UTC, e.g. 2026-11-14T05:12:00+00:00
    summary: str = ""
    region: str = ""
    lang: str = "en"
    weight: float = 1.0         # how much this outlet's coverage counts
    points: int = 0             # upvotes (HN points, Reddit score)
    comments: int = 0
    discussion_url: str = ""    # HN or Reddit thread

    @property
    def uid(self) -> str:
        return f"{self.source}:{self.source_id}"

    @property
    def is_social(self) -> bool:
        return self.source in ("hackernews", "reddit")

    def to_public(self) -> dict:
        d = asdict(self)
        d["id"] = self.uid
        d["summary"] = self.summary[:300]
        d.pop("source_id")
        return d


@dataclass
class Story:
    """Articles that are about the same event, grouped together."""
    id: str
    articles: list[Article]
    headline: str = ""
    url: str = ""
    first_published: str = ""
    topic: str = ""
    topic_weight: float = 0.0
    topics: list[str] = field(default_factory=list)
    score: float = 0.0
    signals: dict = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    first_seen: str = ""
    status: str = ""            # "new", "rising", "steady"
    outlets_yesterday: int = 0
    brief: dict = field(default_factory=dict)   # LLM output, when available
    llm_status: str = ""        # "checked", "unverified", "" (not read)

    def outlet_weights(self) -> dict[str, float]:
        """Distinct publishers, keyed by web domain so two feeds of one site (TechCrunch, TechCrunch AI) count once."""
        from .text import domain
        best: dict[str, list] = {}
        for a in self.articles:
            if a.is_social:
                continue
            d = domain(a.url) or a.outlet
            if d not in best:
                best[d] = [a.outlet, a.weight]
            else:
                best[d][1] = max(best[d][1], a.weight)
        return {name: w for name, w in best.values()}

    @property
    def outlets(self) -> list[str]:
        return sorted(self.outlet_weights())

    @property
    def engagement(self) -> int:
        return sum(a.points + 2 * a.comments for a in self.articles if a.is_social)

    def text(self) -> str:
        return " ".join(f"{a.title}. {a.summary[:400]}" for a in self.articles).lower()

    def to_public(self) -> dict:
        return {
            "id": self.id, "headline": self.headline, "url": self.url, "first_published": self.first_published,
            "topic": self.topic, "topics": self.topics, "score": round(self.score, 1), "signals": self.signals,
            "reasons": self.reasons, "first_seen": self.first_seen, "status": self.status,
            "outlet_count": len(self.outlets), "outlets": self.outlets, "engagement": self.engagement,
            "regions": sorted({a.region for a in self.articles if a.region and not a.is_social}),
            "brief": self.brief, "llm_status": self.llm_status,
            "articles": [a.to_public() for a in sorted(self.articles, key=lambda a: a.published or "z")],
        }
