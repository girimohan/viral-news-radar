from __future__ import annotations

from dataclasses import dataclass, field

from ..models import Article


@dataclass
class SourceResult:
    name: str
    articles: list[Article] = field(default_factory=list)
    ok: bool = True
    error: str = ""
    detail: str = ""
