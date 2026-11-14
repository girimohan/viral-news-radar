"""Group articles about the same event into stories.

Three signals:

1. The same link. A Hacker News or Reddit post that points at a Verge article belongs with that article.
2. The same rare words. Headlines about one event share its distinctive words ("GPT-6", "Supercell",
   "AI Act") even when phrased differently. Words are weighted by how rare they are among today's headlines
   (IDF); words from the feed's snippet count half as much as headline words. An article joins the existing
   story whose centroid it is most similar to (cosine), which stops chains of loosely related headlines
   merging into one blob.
3. The same subject. "Meta cuts 5% of staff" and "Supercell cuts 10% of staff" share most of their words but
   are different stories. Names (OpenAI, GPT-6, Supercell, $2B) are picked out of each headline; when both
   sides name something and they name nothing in common, they are not merged, however similar the wording.

Tuned on tests/fixtures/day.json, a hand-labelled day with hard negatives (python -m newsradar tune).
"""
from __future__ import annotations

import hashlib
import math
import re
from collections import Counter, defaultdict

from .models import Article, Story
from .text import STOPWORDS, canonical_url, terms

# Capitalised, but too general to tell two stories apart.
GENERIC_NAMES = frozenset("""ai ml llm llms eu us usa uk europe european america american china chinese world global
ceo cto ipo api app apps gpu gpus tech big new report exclusive breaking live update watch video opinion analysis
show ask hn tl;dr faq q&a""".split())

_RAW = re.compile(r"[\$€£]?[A-Za-z0-9ÄÖÅäöå][A-Za-z0-9ÄÖÅäöå'’\.\-]*")


def _is_sentence_case(words: list[str]) -> bool:
    """Most outlets write "Supercell to cut staff" (sentence case); some write "Supercell To Cut Staff"."""
    rest = [w for w in words[1:] if w[:1].isalpha() and w.lower() not in STOPWORDS]
    if len(rest) < 2:
        return True
    return sum(w[0].isupper() for w in rest) / len(rest) < 0.6


def _shape_entity(word: str, position: int, sentence_case: bool) -> bool:
    """Does the word look like a name from its shape alone?"""
    core = word.strip("'’.$€£-")
    if not core:
        return False
    if any(c.isdigit() for c in core) and any(c.isalpha() for c in core):
        return True                                  # GPT-6, $2B, Windows 12 -> "12" alone is not
    if any(c.isupper() for c in core[1:]):
        return True                                  # OpenAI, iPad, GPT, EU
    if word[:1] in "$€£" and any(c.isdigit() for c in core):
        return True
    return sentence_case and position > 0 and core[0].isupper()


def names(title: str, learned: set[str] = frozenset(), lower_vocab: set[str] = frozenset()) -> set[str]:
    """Terms of a headline that are names: shaped like one, or known from the day's other text to be one.

    The first word is capitalised anyway, so it counts as a name only when it is known to be one or never
    appears in lower case anywhere in today's headlines and snippets ("Supercell", "Meta", but not "Inside").
    """
    words = _RAW.findall(title or "")
    sc = _is_sentence_case(words)
    out = set()
    for i, w in enumerate(words):
        ts = terms(w)
        if not ts:
            continue
        if _shape_entity(w, i, sc) or ts & learned:
            out |= ts & learned if not _shape_entity(w, i, sc) else ts
        elif i == 0 and w[:1].isupper() and lower_vocab and w.lower().strip("'’") not in lower_vocab:
            out |= ts
    return out


def learn_names(texts: list[str]) -> tuple[set[str], set[str]]:
    """From the day's headlines and snippets: words seen capitalised mid-sentence (names), and every word
    seen in lower case (so a capitalised first word that is never written in lower case is probably a name)."""
    learned, lower = set(), set()
    for t in texts:
        words = _RAW.findall(t or "")
        lower |= {w.strip("'’").removesuffix("'s").removesuffix("’s") for w in words if w[:1].islower()}
        if _is_sentence_case(words):
            for i, w in enumerate(words):
                if _shape_entity(w, i, True):
                    learned |= terms(w)
    return learned - STOPWORDS, lower


def idf_table(docs: list[dict[str, float]]) -> dict[str, float]:
    n = len(docs)
    df = Counter(t for d in docs for t in d)
    return {t: math.log((n + 1) / (c + 1)) + 1 for t, c in df.items()}


def features(a: Article, dek_weight: float) -> dict[str, float]:
    f = {t: 1.0 for t in terms(a.title)}
    if dek_weight:
        for t in terms(" ".join(a.summary.split()[:30])):
            f.setdefault(t, dek_weight)
    return f


class _Group:
    __slots__ = ("members", "tf", "title_tf", "names")

    def __init__(self):
        self.members: list[int] = []
        self.tf: Counter = Counter()        # summed feature weights
        self.title_tf: Counter = Counter()  # headline terms only
        self.names: Counter = Counter()

    def add(self, i: int, f: dict[str, float], nm: set[str]):
        self.members.append(i)
        for t, w in f.items():
            self.tf[t] += w
            if w == 1.0:
                self.title_tf[t] += 1
        self.names.update(nm)

    def absorb(self, other: "_Group"):
        self.members += other.members
        self.tf.update(other.tf)
        self.title_tf.update(other.title_tf)
        self.names.update(other.names)

    def centroid(self, idf: dict[str, float]) -> dict[str, float]:
        n = len(self.members)
        return {t: idf[t] * c / n for t, c in self.tf.items()}

    def main_names(self, common: set[str]) -> set[str]:
        """Names in at least a third of the group's headlines, ignoring generic ones (AI, EU, US...)."""
        n = len(self.members)
        return {t for t, c in self.names.items() if c / n >= 1 / 3 and t not in common}


def similarity(f: dict[str, float], g: _Group, idf: dict[str, float]) -> tuple[float, int]:
    """Cosine between an article's weighted terms and a group's centroid; also the headline terms they share."""
    n = len(g.members)
    dot = 0.0
    for t, w in f.items():
        c = g.tf.get(t)
        if c:
            dot += (idf[t] * w) * (idf[t] * c / n)
    if not dot:
        return 0.0, 0
    na = math.sqrt(sum((idf[t] * w) ** 2 for t, w in f.items()))
    ng = math.sqrt(sum((idf[t] * c / n) ** 2 for t, c in g.tf.items()))
    shared = sum(1 for t, w in f.items() if w == 1.0 and t in g.title_tf)
    return dot / (na * ng), shared


def cluster(articles: list[Article], threshold: float = 0.2, min_shared: int = 1, dek_weight: float = 0.5,
            nameless_penalty: float = 0.1) -> list[Story]:
    """Group articles into stories. See the module docstring for how."""
    order = sorted(range(len(articles)), key=lambda i: articles[i].published or "9999")
    feats = [features(a, dek_weight) for a in articles]
    idf = idf_table(feats)
    learned, lower = learn_names([a.title for a in articles] + [a.summary[:400] for a in articles])
    nms = [names(a.title, learned, lower) for a in articles]
    common = set(GENERIC_NAMES)

    groups: list[_Group] = []
    by_url: dict[str, int] = {}
    by_term: dict[str, set[int]] = defaultdict(set)
    for i in order:
        a, f = articles[i], feats[i]
        url = canonical_url(a.url)
        gi = by_url.get(url)
        if gi is None and f:
            mine = nms[i] - common
            best, best_sim = None, threshold
            for c in set().union(*(by_term[t] for t in f if t in by_term)):
                g = groups[c]
                theirs = g.main_names(common)
                if mine and theirs and not (mine & theirs):
                    continue                       # different subjects
                sim, shared = similarity(f, g, idf)
                if not (mine and theirs):
                    sim -= nameless_penalty        # a vague headline needs stronger wording overlap
                if shared >= min_shared and sim >= best_sim:
                    best, best_sim = c, sim
            gi = best
        if gi is None:
            gi = len(groups)
            groups.append(_Group())
        groups[gi].add(i, f, nms[i])
        by_url[url] = gi
        for t in f:
            by_term[t].add(gi)

    groups = merge_pass(groups, idf, common, threshold)
    stories = []
    for g in groups:
        members = [articles[i] for i in g.members]
        cent = {k: similarity(feats[i], g, idf)[0] for k, i in enumerate(g.members)}
        stories.append(build_story(members, cent))
    return stories


def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
    dot = sum(v * b[t] for t, v in a.items() if t in b)
    if not dot:
        return 0.0
    return dot / (math.sqrt(sum(v * v for v in a.values())) * math.sqrt(sum(v * v for v in b.values())))


def merge_pass(groups: list[_Group], idf: dict[str, float], common: set[str], threshold: float) -> list[_Group]:
    """The first pass is greedy, so its result depends on arrival order: an early headline phrased differently
    can start a story of its own. Merge groups that name the same thing and whose centroids are similar."""
    merged = True
    while merged:
        merged = False
        by_name: dict[str, list[int]] = defaultdict(list)
        mains = [g.main_names(common) for g in groups]
        for k, nm in enumerate(mains):
            for t in nm:
                by_name[t].append(k)
        cents = [g.centroid(idf) for g in groups]
        gone: set[int] = set()
        for k in range(len(groups)):
            if k in gone or not mains[k]:
                continue
            for j in sorted({j for t in mains[k] for j in by_name[t] if j > k and j not in gone}):
                if _cosine(cents[k], cents[j]) >= threshold:
                    groups[k].absorb(groups[j])
                    gone.add(j)
                    merged = True
            cents[k] = groups[k].centroid(idf)
        groups = [g for k, g in enumerate(groups) if k not in gone]
    return groups


def build_story(members: list[Article], centrality: dict[int, float] | None = None) -> Story:
    """Pick the headline and link that best represent the group: the most central article from a publisher
    (falling back to a social post), preferring heavier outlets, then the earliest."""
    centrality = centrality or {}
    ranked = sorted(range(len(members)), key=lambda k: (members[k].is_social, -round(centrality.get(k, 0), 1),
                                                        -members[k].weight, members[k].published or "9999"))
    rep = members[ranked[0]]
    dated = sorted(a.published for a in members if a.published)
    uid = hashlib.sha1(min(a.uid for a in members).encode()).hexdigest()[:10]
    return Story(id=uid, articles=members, headline=rep.title, url=rep.url, first_published=dated[0] if dated else "")


def story_terms(s: Story, top: int = 14) -> list[str]:
    """The words most of a story's headlines share, for matching it against earlier days' stories."""
    c = Counter(t for a in s.articles for t in terms(a.title))
    return [t for t, _ in c.most_common(top)]


# ------------------------------------------------------------------ tuning
def load_labelled(path) -> tuple[list[Article], list[str]]:
    import json
    items = json.loads(open(path, encoding="utf-8").read())["items"]
    arts, labels = [], []
    for i, (label, outlet, title, dek) in enumerate(items):
        social = outlet == "Hacker News" or outlet.startswith("r/")
        arts.append(Article(source="hackernews" if social else "feed", source_id=str(i), outlet=outlet, title=title,
                            url=f"https://example.com/{i}", published=f"2026-11-14T{i % 24:02d}:00:00+00:00",
                            summary=dek))
        labels.append(label if label != "-" else f"solo{i}")
    return arts, labels


def pair_scores(stories: list[Story], arts: list[Article], labels: list[str]) -> dict:
    """Pairwise precision/recall of a clustering against hand labels."""
    import itertools
    where = {id(a): k for k, s in enumerate(stories) for a in s.articles}
    tp = fp = fn = 0
    for i, j in itertools.combinations(range(len(arts)), 2):
        same_t, same_p = labels[i] == labels[j], where[id(arts[i])] == where[id(arts[j])]
        tp += same_t and same_p
        fp += (not same_t) and same_p
        fn += same_t and not same_p
    p = tp / (tp + fp) if tp + fp else 1.0
    r = tp / (tp + fn) if tp + fn else 1.0
    return {"precision": round(p, 3), "recall": round(r, 3), "f1": round(2 * p * r / (p + r), 3) if p + r else 0.0}
