from __future__ import annotations

import html
import re
from functools import lru_cache
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")
_WORD = re.compile(r"[a-z0-9äöåü][a-z0-9äöåü'\.\-]*[a-z0-9äöåü]|[a-z0-9äöåü]")

# Common English and Finnish words that say nothing about which story a headline is about.
STOPWORDS = frozenset("""
a about above after again against all also am an and any are as at be because been before being below between both
but by can could did do does doing down during each few for from further had has have having he her here hers him his
how i if in into is it its itself just like me more most my new no nor not now of off on once only or other our out
over own same says say said she should so some such than that the their them then there these they this those through
to too under until up very was we were what when where which while who whom why will with would you your yours
amid over vs via after gets get got make makes made big first last next week today yesterday report reports reportedly
year years day days time times may might new news update updates live latest here's heres what's whats how why
ja on ei se että oli ovat kun mutta myös tai joka jotka nyt uusi uuden vuonna jo vielä kuin sen hän he me te
""".split())

_TRACKING = re.compile(r"^(utm_|fbclid|gclid|mc_|ref$|ref_|cmp$|cmpid|ito$|at_|guccounter|ncid|sr_share|source$)")


def strip_html(s: str | None) -> str:
    """HTML (possibly double-escaped, as some feeds send it) to plain text."""
    if not s:
        return ""
    s = html.unescape(html.unescape(s))
    return _WS.sub(" ", _TAG.sub(" ", s)).strip()


@lru_cache(maxsize=4096)
def keyword_re(keyword: str) -> re.Pattern:
    """Match a keyword only as a whole word or phrase ("ai" must not match "said" or "Thailand")."""
    body = re.escape(keyword.lower().strip()).replace(r"\ ", r"[\s\-/]+")
    return re.compile(rf"(?<![a-z0-9äöå]){body}(?![a-z0-9äöå])")


def has_keyword(text: str, keyword: str) -> bool:
    return bool(keyword_re(keyword).search(text))


def _stem(w: str) -> str:
    """Very light stemming so "launches"/"launched"/"launch" and "models"/"model" meet."""
    w = w.strip(".'-")
    if len(w) > 5 and w.endswith("ing"):
        return w[:-3]
    if len(w) > 4 and w.endswith("ed"):
        return w[:-2]
    if len(w) > 4 and w.endswith("es") and not w.endswith("ses"):
        return w[:-2]
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    if w.endswith("'s"):
        return w[:-2]
    return w


def terms(text: str) -> set[str]:
    """Content words of a headline, lower-cased and lightly stemmed, stopwords removed."""
    out = set()
    for w in _WORD.findall((text or "").lower().replace("’", "'")):
        w = w.removesuffix("'s")
        if w in STOPWORDS or len(w) < 2 or w.isdigit() and len(w) < 3:
            continue
        s = _stem(w)
        if s and s not in STOPWORDS:
            out.add(s)
    return out


def canonical_url(url: str) -> str:
    """Strip tracking parameters, fragments, "www." and trailing slashes, so the same article matches itself."""
    try:
        p = urlsplit((url or "").strip())
    except ValueError:
        return url or ""
    host = p.netloc.lower().removeprefix("www.").removeprefix("m.")
    q = urlencode([(k, v) for k, v in parse_qsl(p.query) if not _TRACKING.match(k.lower())])
    return urlunsplit(("https", host, p.path.rstrip("/") or "/", q, ""))


def domain(url: str) -> str:
    try:
        return urlsplit(url).netloc.lower().removeprefix("www.")
    except ValueError:
        return ""
