# Viral News Radar

Finds the news stories that many outlets and readers are picking up today, on the topics you follow, and
sends them to you every morning. Built on the same pattern as
[remote-job-radar](https://github.com/girimohan/remote-job-radar), aimed at sourcing stories for a newsletter
such as AIFinPost.

"Viral" here means something you can measure: how many different outlets ran the story, how much discussion it
drew on Hacker News and Reddit, how fast that grew, and how fresh it is.

## What you get

| | |
|---|---|
| **Dashboard** | `docs/index.html`, served by GitHub Pages. Stories ranked by a 0-100 virality score, with topic and "new / rising / Finland" filters, the reasons behind each score, every outlet's own article, and a **Save for newsletter** list you can copy out as Markdown. |
| **Morning digest** | The top 15 stories (plus Finnish and Nordic ones) arrive as a GitHub issue each morning, which GitHub emails to you. Also saved in `digests/latest.md`. |
| **Data** | `docs/news.json` holds today's stories. `data/seen.json` remembers stories across days so it can say what is new and what is rising. |

## How it works

```
collect  ->  last 48 h  ->  drop noise  ->  cluster into stories  ->  match to earlier days  ->  score  ->  brief  ->  dashboard + digest
```

1. **Collect.** About 20 publisher RSS/Atom feeds (Yle, Helsingin Sanomat, Helsinki Times, ArcticStartup, Tivi,
   TechCrunch, The Verge, Ars Technica, Wired, MIT Technology Review, The Guardian, BBC, Sifted, Tech.eu,
   Euractiv, OpenAI, Google AI, Hugging Face...), Hacker News stories with 40+ points (public Algolia API), and
   optionally the day's top posts in chosen subreddits (official Reddit API).
2. **Cluster** (`newsradar/cluster.py`). Articles about the same event are grouped into one story using
   - the same link (a Hacker News post pointing at a Verge article joins it),
   - the same rare words, weighted by IDF over the day's headlines, with feed snippets counting half,
     compared with each story's centroid so loosely related headlines do not chain together,
   - the same subject: names (OpenAI, GPT-6, Supercell, $2B) are picked out of each headline, and two sides
     that both name something but nothing in common are never merged. This is what keeps "Meta cuts 5% of
     staff" apart from "Supercell cuts 10% of staff",
   - a second pass that merges groups the first, order-dependent pass left split.
3. **Match to earlier days.** Each story is matched to the ledger by shared links or key words, so it keeps
   its id across days. Stories are marked **new**, **rising** (more outlets than on the last day it was seen)
   or steady.
4. **Score (0-100)** (`newsradar/scoring.py`). Coverage (distinct outlets, weighted), engagement (points and
   comments) and velocity (engagement per hour) combine as `1 - (1-a)(1-b)(1-c)`, so either wide coverage or
   strong social buzz alone can rank a story, and both together rank it highest. Freshness and the topic's
   weight multiply the result. Two feeds of one site count as one outlet.
5. **Brief** (`newsradar/llm.py`, optional). Claude (Haiku 5.5) reads the headlines and snippets of the top 25
   stories and writes a neutral headline, a two-sentence summary, why it matters, a newsletter angle and how it
   relates to Finland.

## Set up (once)

1. **Create the repository** `viral-news-radar` on GitHub and push this folder to it (commands below).
2. **Run it:** Actions tab -> *Daily news radar* -> *Run workflow*. After that it runs by itself every morning
   (about 06:11 Helsinki time in winter, 07:11 in summer, with a backup run later in the morning).
3. **Turn on the dashboard:** Settings -> Pages -> *Deploy from a branch* -> branch `main`, folder `/docs`.
   It appears at `https://<your-user>.github.io/viral-news-radar/`.
4. **Get the digest by email:** Watch -> All activity on the repository.

```bash
cd viral-news-radar
git init && git add . && git commit -m "Viral news radar"
git branch -M main
git remote add origin https://github.com/<your-user>/viral-news-radar.git
git push -u origin main
```

The dashboard shows demo data (from `tests/fixtures/day.json`) until the first run replaces it.

### Optional: briefs and Reddit

- **Briefs:** Settings -> Secrets and variables -> Actions -> New repository secret `ANTHROPIC_API_KEY`
  (the same key as remote-job-radar works). About 25 short calls a day, roughly a cent; stories already briefed
  are cached and only re-read when new outlets join them.
- **Reddit:** create a "script" app at reddit.com/prefs/apps and add `REDDIT_CLIENT_ID` and
  `REDDIT_CLIENT_SECRET` secrets. Without them the Reddit source is skipped, which is fine.

## Change what it looks for

Everything is in `config.yaml`:

- `topics`: what you care about, with keywords and a weight. Stories that match no topic are dropped unless at
  least `untopical_min_outlets` outlets cover them (big breaking news).
- `feeds`: add or remove any RSS/Atom feed; `weight` says how much that outlet's coverage counts.
- `exclude_title`: headline words that drop a story (deals, horoscopes, reviews...).
- `scoring` and `min_score`: how the signals mix and how strict the list is.
- `clustering`: how readily headlines merge into one story. Check a change with `python -m newsradar tune`.
- `llm.audience`: who the briefs and angles are written for.

## Run locally

```bash
pip install -r requirements-dev.txt
python -m pytest -q
python -m newsradar run                      # full run, under a minute
python -m newsradar run --sources hackernews # quick partial run
python -m newsradar tune                     # clustering precision/recall against hand labels
python -m http.server -d docs                # dashboard at http://localhost:8000
```

## Measuring the clustering

Clustering decides the coverage count, so a wrong merge makes a story look more viral than it is.
`tests/fixtures/day.json` is a hand-labelled day of 36 headlines with deliberate hard cases: the same event
phrased five ways, and different companies doing the same kind of thing (two "cuts staff", three "raises",
several "launches a model"). `python -m newsradar tune` reports pairwise precision and recall at the current
settings and across a grid. The current settings get both right on that set, and a wide range of thresholds
around them do too, so the result is not one lucky setting. The set is small and was written by hand, though:
add a few real days (copy headlines from `docs/news.json` and label which belong together) before trusting
the numbers further.

## Limits

- Coverage is only as wide as the feed list. A story that no listed outlet runs will not be found.
- Briefs are written from headlines and feed snippets, not full articles. Each must quote the text it relied
  on word for word, or it is discarded, but it can still be wrong. Read the source before you publish.
- Finnish- and English-language coverage of the same event are counted as separate stories (the words differ).
- Google News is not used: its `robots.txt` disallows automated access.

## Good citizenship

Only public feeds and documented APIs. The collector identifies itself, runs about twice a day, uses 8
concurrent requests and backs off when rate-limited. Every story links to the outlets' own articles; the
dashboard shows headlines and short snippets only.
