from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml

from . import pipeline, report

ROOT = Path(__file__).resolve().parent.parent


def cmd_run(args) -> int:
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    only = set(args.sources.split(",")) if args.sources else None
    try:
        payload = pipeline.run(cfg, Path(args.out), only=only, ledger_path=ROOT / "data" / "seen.json")
    except RuntimeError as e:
        print(e, file=sys.stderr)  # nothing is overwritten when every source is down
        return 1
    text = report.digest(payload, args.dashboard_url)
    d = Path(args.digest)
    d.parent.mkdir(parents=True, exist_ok=True)
    d.write_text(text, encoding="utf-8")

    f = payload["funnel"]
    print(f"collected {f['collected']:,} articles -> {f['articles_in_window']:,} in window -> {f['stories']:,} stories "
          f"-> {f['shown']} shown (new today: {payload['new_today']})")
    for s in payload["sources"]:
        print(f"  {s['name']:11} {'ok ' if s['ok'] else 'ERR'} fetched {s['fetched']:>5,} in stories {s['in_stories']:>3}  {s['error'] or s['detail']}")
    u = payload.get("llm", {})
    if u.get("enabled"):
        print(f"  LLM {u['model']}: {u['calls']} new briefs, {u['cached']} cached, {u['failed']} failed, "
              f"{u['unverified']} discarded (quote not found), ${u['cost_usd']:.4f}")
    else:
        print("  LLM: off (set ANTHROPIC_API_KEY to turn on briefs)")
    for i, s in enumerate(payload["stories"][:10], 1):
        print(f"  {i:>2}. {s['score']:>5.1f}  {s['headline'][:90]}")
    gh = os.environ.get("GITHUB_OUTPUT")
    if gh:
        with open(gh, "a", encoding="utf-8") as fh:
            fh.write(f"new_today={payload['new_today']}\nshown={f['shown']}\nfirst_run={'true' if payload['first_run'] else 'false'}\n")
    return 0


def cmd_tune(args) -> int:
    """Score the clustering against hand-labelled headlines, at the configured settings and across a grid."""
    import itertools
    from . import cluster as clu
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")).get("clustering", {})
    arts, labels = clu.load_labelled(args.labels)
    cur = (cfg.get("threshold", 0.18), cfg.get("min_shared_terms", 1), cfg.get("dek_weight", 0.5), cfg.get("nameless_penalty", 0.1))
    m = clu.pair_scores(clu.cluster(arts, *cur), arts, labels)
    print(f"{len(arts)} labelled headlines, {len(set(labels))} stories")
    print(f"current settings {cur}: precision {m['precision']}, recall {m['recall']}, F1 {m['f1']}")
    rows = []
    for thr, ms, dw, pen in itertools.product([x / 100 for x in range(10, 42, 2)], (1, 2), (0, 0.5, 1.0), (0, 0.1, 0.2)):
        r = clu.pair_scores(clu.cluster(arts, thr, ms, dw, pen), arts, labels)
        rows.append((r["f1"], r["precision"], r["recall"], thr, ms, dw, pen))
    rows.sort(reverse=True)
    print("best on this set (f1, precision, recall, threshold, min_shared_terms, dek_weight, nameless_penalty):")
    for r in rows[:8]:
        print("  ", r)
    print("Precision matters more than recall here: a wrong merge makes a story look more viral than it is.")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="newsradar", description="Find today's most widely shared stories on your topics.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="collect, cluster, score -> docs/news.json + digests/latest.md")
    r.add_argument("--config", default=str(ROOT / "config.yaml"))
    r.add_argument("--out", default=str(ROOT / "docs" / "news.json"))
    r.add_argument("--digest", default=str(ROOT / "digests" / "latest.md"))
    r.add_argument("--sources", help="comma-separated subset: feeds,hackernews,reddit")
    r.add_argument("--dashboard-url", default=os.environ.get("DASHBOARD_URL", ""))
    r.set_defaults(fn=cmd_run)
    t = sub.add_parser("tune", help="check the story clustering against hand-labelled headlines")
    t.add_argument("--config", default=str(ROOT / "config.yaml"))
    t.add_argument("--labels", default=str(ROOT / "tests" / "fixtures" / "day.json"))
    t.set_defaults(fn=cmd_tune)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
