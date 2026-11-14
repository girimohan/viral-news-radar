import datetime as dt
import json
from pathlib import Path

import yaml

from newsradar import cluster as cluster_mod
from newsradar import llm, pipeline, report
from newsradar.cluster import cluster
from newsradar.models import Article
from newsradar.sources import SourceResult, feeds, hackernews, reddit
from newsradar.text import canonical_url, terms

ROOT = Path(__file__).resolve().parent.parent
FIX = Path(__file__).parent / "fixtures"
CFG = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
NOW = dt.datetime(2026, 11, 14, 7, 0, tzinfo=dt.timezone.utc)


def art(outlet, title, url, hours_ago=3, summary="", source="feed", points=0, comments=0, weight=0.9):
    t = (NOW - dt.timedelta(hours=hours_ago)).isoformat(timespec="seconds")
    return Article(source=source, source_id=f"{outlet}:{title[:20]}", outlet=outlet, title=title, url=url,
                   published=t, summary=summary, weight=weight, points=points, comments=comments,
                   discussion_url="https://news.ycombinator.com/item?id=1" if source == "hackernews" else "")


DOMAINS = {"The Verge": "theverge.com", "TechCrunch": "techcrunch.com", "Ars Technica": "arstechnica.com",
           "Wired": "wired.com", "BBC Technology": "bbc.co.uk", "Engadget": "engadget.com", "Yle News": "yle.fi",
           "Helsinki Times": "helsinkitimes.fi", "Sifted": "sifted.eu", "The Information": "theinformation.com",
           "Euractiv Tech": "euractiv.com", "The Guardian Tech": "theguardian.com", "MIT Technology Review":
           "technologyreview.com", "Tech.eu": "tech.eu"}


def day_one():
    """The hand-labelled day in fixtures/day.json, as articles published in the last few hours, plus social
    engagement on the GPT-6 story and one story from last week."""
    arts, _ = cluster_mod.load_labelled(FIX / "day.json")
    out = []
    for i, a in enumerate(arts):
        if a.outlet == "Hacker News" and a.title == "GPT-6":
            out.append(art("Hacker News", "GPT-6", "https://www.theverge.com/0", 2, source="hackernews", points=900, comments=600))
            continue
        if a.is_social or a.outlet.startswith("r/"):
            continue
        out.append(art(a.outlet, a.title, f"https://{DOMAINS[a.outlet]}/{i}", 1 + i % 9, a.summary,
                       weight=1.0 if a.outlet == "Yle News" else 0.9))
    out.append(art("The Verge", "Old story from last week about OpenAI", "https://www.theverge.com/old", 24 * 5))
    return out


def test_parse_rss_and_atom():
    v = feeds.parse_feed((FIX / "verge.xml").read_bytes(), {"name": "The Verge", "weight": 0.9, "region": "International"})
    assert [a.title for a in v][0] == "OpenAI launches GPT-6 with a much longer memory"
    assert v[0].published == "2026-11-14T03:10:00+00:00"
    assert "remember entire projects" in v[0].summary and "<p>" not in v[0].summary
    assert v[2].published == ""  # undated items are kept by the parser, dropped by the pipeline
    a = feeds.parse_feed((FIX / "atom.xml").read_bytes(), {"name": "Example"})
    assert a[0].url == "https://example.fi/supercell-layoffs"
    assert a[0].published == "2026-11-14T04:00:00+00:00"


def test_hackernews_and_reddit_parsers():
    hn = hackernews.parse({"hits": [{"objectID": "9", "title": "GPT-6", "url": "https://x.com/a", "points": 500,
                                      "num_comments": 300, "created_at_i": 1789000000},
                                     {"objectID": "8", "title": "small", "points": 3}]}, min_points=40)
    assert len(hn) == 1 and hn[0].points == 500 and hn[0].discussion_url.endswith("id=9")
    rd = reddit.parse({"data": {"children": [
        {"data": {"id": "a", "title": "GPT-6 released", "score": 4000, "num_comments": 900, "permalink": "/r/x/a",
                  "url": "https://openai.com/gpt6", "created_utc": 1789000000}},
        {"data": {"id": "b", "title": "pinned", "score": 9999, "stickied": True}}]}}, "OpenAI", 150)
    assert len(rd) == 1 and rd[0].outlet == "r/OpenAI" and rd[0].url == "https://openai.com/gpt6"


def test_reddit_skipped_without_credentials(monkeypatch):
    monkeypatch.delenv("REDDIT_CLIENT_ID", raising=False)
    r = reddit.fetch(["artificial"])
    assert r.ok and not r.articles and "skipped" in r.detail


def test_text_helpers():
    assert canonical_url("https://www.theverge.com/gpt-6/?utm_source=rss#top") == canonical_url("https://theverge.com/gpt-6")
    assert {"openai", "launch", "gpt-6", "memory"} <= terms("OpenAI launches GPT-6 with a much longer memory")


def test_clustering_matches_hand_labels():
    arts, labels = cluster_mod.load_labelled(FIX / "day.json")
    m = cluster_mod.pair_scores(cluster(arts, **{k: v for k, v in CFG["clustering"].items() if k != "min_shared_terms"},
                                        min_shared=CFG["clustering"]["min_shared_terms"]), arts, labels)
    assert m["precision"] >= 0.95 and m["recall"] >= 0.9, m


def test_same_kind_of_event_different_company_is_not_merged():
    stories = cluster(day_one())
    meta = next(s for s in stories if any(a.title.startswith("Meta cuts") for a in s.articles))
    assert len(meta.articles) == 1
    sc = next(s for s in stories if any(a.title.startswith("Supercell to cut") for a in s.articles))
    assert len(sc.articles) == 3
    gpt = max(stories, key=lambda s: len(s.articles))
    assert "Hacker News" in {a.outlet for a in gpt.articles}   # joined by its link


def test_two_feeds_of_one_site_count_once():
    a = [art("TechCrunch", "Mistral raises $2B at $20B valuation", "https://techcrunch.com/mistral"),
         art("TechCrunch AI", "Mistral raises $2B at $20B valuation", "https://techcrunch.com/mistral?utm_medium=ai")]
    s = cluster(a)
    assert len(s) == 1 and s[0].outlets == ["TechCrunch"]


def run_day(tmp_path, arts, now=NOW, reader=None):
    return pipeline.run(CFG, tmp_path / "docs" / "news.json", now=now, ledger_path=tmp_path / "data" / "seen.json",
                        reader=reader or llm.Reader(api_key=""), results=[SourceResult("feeds", arts)])


def test_pipeline_ranks_and_filters(tmp_path):
    p = run_day(tmp_path, day_one())
    heads = [s["headline"] for s in p["stories"]]
    assert "GPT-6" in heads[0]
    assert any("Supercell" in h for h in heads)
    assert not any("bus timetable" in h for h in heads)      # off-topic
    assert not any("Tesla recalls" in h for h in heads)      # on no topic list, one outlet
    assert not any("Old story" in h for h in heads)          # outside the window
    top = p["stories"][0]
    assert top["outlet_count"] == 5 and top["engagement"] == 2100
    assert any("HN 900 points" in r for r in top["reasons"])
    assert p["first_run"] and p["new_today"] == 0
    md = report.digest(p, "https://example.github.io/viral-news-radar/")
    assert "GPT-6" in md and "Open the dashboard" in md
    json.loads((tmp_path / "docs" / "news.json").read_text())


def test_second_day_marks_new_and_rising(tmp_path):
    run_day(tmp_path, day_one())
    tomorrow = NOW + dt.timedelta(days=1)
    arts = [a for a in day_one() if "Supercell" in a.title]
    for a in arts:
        a.published = (tomorrow - dt.timedelta(hours=20)).isoformat(timespec="seconds")
    arts.append(art("Tivi", "Supercell staff cuts hit Helsinki gaming scene", "https://tivi.fi/supercell", 0))
    arts[-1].published = (tomorrow - dt.timedelta(hours=2)).isoformat(timespec="seconds")
    arts.append(art("Hacker News", "Hugging Face releases open-weights SmolLM 4", "https://huggingface.co/blog/smollm4",
                    source="hackernews", points=700, comments=300))
    arts[-1].published = (tomorrow - dt.timedelta(hours=3)).isoformat(timespec="seconds")
    p = run_day(tmp_path, arts, now=tomorrow)
    st = {s["headline"]: s for s in p["stories"]}
    sc = next(s for h, s in st.items() if "Supercell" in h)
    assert sc["status"] == "rising" and sc["first_seen"] == NOW.date().isoformat()
    nv = next(s for h, s in st.items() if "SmolLM" in h)
    assert nv["status"] == "new" and p["new_today"] == 1


def test_every_source_down_raises(tmp_path):
    import pytest
    with pytest.raises(RuntimeError):
        pipeline.run(CFG, tmp_path / "news.json", now=NOW, ledger_path=tmp_path / "seen.json",
                     results=[SourceResult("feeds", ok=False, error="every feed failed")])


def test_llm_brief_kept_only_with_verbatim_evidence(tmp_path):
    def fake(system, user, schema, **kw):
        good = "GPT-6" in user
        return ({"headline": "OpenAI releases GPT-6", "summary": "OpenAI released GPT-6.", "why_it_matters": "x",
                 "angle": "y", "finland_link": "none", "category": "models", "confidence": "high",
                 "evidence": "OpenAI launches GPT-6 with a much longer memory" if good else "a sentence nobody wrote"},
                {"input_tokens": 1000, "output_tokens": 150})
    reader = llm.Reader(api_key="test", call=fake, cache_path=tmp_path / "cache.json")
    p = run_day(tmp_path, day_one(), reader=reader)
    top = p["stories"][0]
    assert top["llm_status"] == "checked" and top["brief"]["headline"] == "OpenAI releases GPT-6"
    other = next(s for s in p["stories"] if "Supercell" in s["headline"])
    assert other["llm_status"] == "unverified" and other["brief"] == {}
    assert p["llm"]["calls"] >= 2 and p["llm"]["unverified"] >= 1
    # second run is served from the cache
    reader2 = llm.Reader(api_key="test", call=fake, cache_path=tmp_path / "cache.json")
    p2 = run_day(tmp_path, day_one(), reader=reader2)
    assert p2["llm"]["calls"] == 0 and p2["llm"]["cached"] >= 2
