"""
merge the per-month CSVs produced by reddit-dump-extractor
(RC_YYYY-MM.csv = comments, RS_YYYY-MM.csv = submissions) into one .csv file.

Pipeline
 1. Read raw CSV as strings (the extractor casts many columns to str, so NaN
    shows up as the literal text "nan" / "None" and is converted back to real NA).
 2. Filter subreddits to the project list and attach a topic `category`.
 3. Parse `created_utc`, keep only the study window (May 2025 - Aug 2026).
 4. Remove duplicates by (type, id) across all files
 5. Normalise authors ("[deleted]" -> NA), drop known bots (AutoModerator...) and
    moderator/admin/stickied posts.
 6. Build one `text` field (comment body, or submission title + selftext),
    drop "[removed]"/"[deleted]"/empty rows, fix HTML entities, whitespace, unicode.
 7. Flag `mentions_ai` (keyword regex). Optionally DROP non-AI rows in chosen
    categories (--require-ai-in), e.g. generic mental-health subs.
 8. Optional THREAD-level AI flag (--thread-flags, keeps all rows) or filter (--thread-ai-in): keeps whole threads (post + all
    comments) where the post or any comment mentions AI
 9. Append to a single output CSV + write a JSON report with all counts.
"""

from __future__ import annotations

import argparse
import html
import json
import logging
import re
import sys
import unicodedata
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
-o
import numpy as np
import pandas as pd

# PROJECT CONFIG 

SUBREDDITS: dict[str, tuple[str, str]] = {
    # AI partners
    "aipartners":          ("aipartners",          "ai_partners"),
    "myboyfriendisai":     ("MyBoyfriendIsAI",     "ai_partners"),
    "mygirlfriendisai":    ("MyGirlfriendIsAI",    "ai_partners"),
    # AI sentience
    "artificialsentience": ("ArtificialSentience", "ai_sentience"),
    # AI as mental-health substitute
    "mentalhealth":        ("mentalhealth",        "mental_health"),
    "therapy":             ("therapy",             "mental_health"),
    # General AI communities (baseline)
    "artificial":          ("artificial",          "general_ai"),
    "singularity":         ("singularity",         "general_ai"),
    "artificialinteligence": ("ArtificialInteligence", "general_ai"),
    "beyondthepromptai":    ("BeyondThePromptAI",    "general_ai"),
}

AI_KEYWORDS = [
    r"ai", r"a\.i\.", r"artificial intelligence", r"chat-?gpt", r"gpt-?\w*",
    r"llms?", r"language models?", r"chatbots?", r"openai", r"anthropic",
    r"claude", r"gemini", r"copilot", r"grok", r"deepseek", r"replika",
    r"character\.?ai", r"c\.ai", r"kindroid", r"nomi\.ai", r"machine learning",
]
AI_REGEX = r"(?<![\w])(?:" + "|".join(AI_KEYWORDS) + r")(?![\w])"

BOT_NAMES = {
    "automoderator", "sneakpeekbot", "remindmebot", "repostsleuthbot",
    "wikitextbot", "gifreversingbot", "b0trank", "converter-bot",
    "totesmessenger", "imguralbumbot", "savevideo", "vredditdownloader",
}
BOT_REGEX = r"(?i)(?:[-_]bot$|^bot[-_]|_bot_)"

NA_STRINGS = {"", "nan", "none", "null", "<na>"}
REMOVED_MARKERS = {"[removed]", "[deleted]", "[removed by reddit]"}

FILE_RE = re.compile(r"^(RC|RS)_(\d{4})-(\d{2})\.csv(\.gz)?$")
KIND = {"RC": "comment", "RS": "submission"}

RAW_COLUMNS = {
    "comment": ["id", "subreddit", "author", "created_utc", "score",
                "body", "link_id", "parent_id", "permalink",
                "distinguished", "stickied", "is_submitter", "total_awards_received"],
    "submission": ["id", "subreddit", "author", "created_utc", "score",
                   "title", "selftext", "num_comments", "over_18", "link_flair_text",
                   "permalink", "upvote_ratio", "num_crossposts", "total_awards_received",
                   "subreddit_subscribers", "is_self", "is_video", "is_gallery",
                   "crosspost_parent", "domain", "url", "removed_by_category",
                   "distinguished", "stickied"],
}
NUMERIC_COLS = ["score", "num_comments", "upvote_ratio", "num_crossposts",
                "total_awards_received", "subreddit_subscribers"]
BOOL_COLS = ["over_18", "is_self", "is_video", "is_gallery", "stickied", "is_submitter"]
MOD_LEVELS = {"moderator", "admin"} 

OUTPUT_COLUMNS = [
    "id", "type", "subreddit", "category", "author",
    "created_utc", "created_dt", "month",
    "score", "upvote_ratio", "title", "text", "n_words", "mentions_ai",
    "thread_id", "parent_id", "is_submitter",
    "num_comments", "num_crossposts", "total_awards_received", "subreddit_subscribers",
    "over_18", "is_self", "is_video", "is_gallery", "is_crosspost", "domain", "url",
    "link_flair_text", "removed_by_category", "distinguished", "stickied", "permalink",
]

log = logging.getLogger("preprocess")

# HELPERS

def clean_str(s: pd.Series) -> pd.Series:
    """strip whitespace and turn 'nan'/'None'/'' into NA."""
    s = s.astype("string").str.strip()
    return s.mask(s.str.lower().isin(NA_STRINGS))


def normalize_text(s: pd.Series) -> pd.Series:
    """HTML-unescape, NFC-normalise, strip control chars, tidy whitespace."""
    s = s.astype("string[python]")
    has_entity = s.str.contains(r"&(?:amp|lt|gt|quot|apos|#\d+|#x[0-9a-fA-F]+);", na=False)
    if has_entity.any():
        s = s.mask(has_entity, s.map(html.unescape, na_action="ignore"))
    non_ascii = ~s.map(lambda x: x.isascii() if isinstance(x, str) else True).astype(bool)
    if non_ascii.any():
        s = s.mask(non_ascii, s.map(lambda x: unicodedata.normalize("NFC", x), na_action="ignore"))
    s = (s.str.replace("\r\n", "\n", regex=False)
          .str.replace("\r", "\n", regex=False)
          .str.replace(r"[\u200b-\u200d\u2060\ufeff\x00-\x08\x0b\x0c\x0e-\x1f]", "", regex=True)
          .str.replace(r"[ \t\u00a0]+", " ", regex=True)
          .str.replace(r" ?\n ?", "\n", regex=True)
          .str.replace(r"\n{3,}", "\n\n", regex=True)
          .str.strip())
    return s.mask(s == "")

def blank_removed(s: pd.Series) -> pd.Series:
    return s.mask(s.str.lower().isin(REMOVED_MARKERS))

def to_bool(s: pd.Series) -> pd.Series:
    m = s.astype("string").str.strip().str.lower().map(
        {"true": True, "false": False, "1": True, "0": False})
    return m.astype("boolean")

def hash_rows(*cols: pd.Series) -> np.ndarray:
    joined = cols[0].astype("string")
    for c in cols[1:]:
        joined = joined + "\x1f" + c.astype("string").fillna("")
    return pd.util.hash_pandas_object(joined, index=False).to_numpy()


def discover_files(inputs: list[str]) -> list[tuple[str, str, Path]]:
    found = []
    for item in inputs:
        p = Path(item)
        cands = [p] if p.is_file() else sorted(p.rglob("*")) if p.is_dir() else []
        if not cands:
            log.warning("Input not found or empty: %s", item)
        for f in cands:
            m = FILE_RE.match(f.name)
            if m:
                found.append((f"{m.group(2)}-{m.group(3)}", KIND[m.group(1)], f))
    return sorted(found, key=lambda t: (t[0], t[1], str(t[2])))


# CLEAN ONE CHUNK

class Ctx:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.sub_map = {k: v for k, v in SUBREDDITS.items()
                        if args.subreddits is None or k in args.subreddits}
        self.start_ts = int(datetime.strptime(args.start, "%Y-%m-%d")
                            .replace(tzinfo=timezone.utc).timestamp())
        end = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        self.end_ts = int((end + timedelta(days=1)).timestamp())
        self.require_ai = set(args.require_ai_in or [])
        self.seen_ids: set[int] = set()
        self.seen_text: set[int] = set()
        self.dropped: dict[str, int] = defaultdict(int)
        self.rows_read = 0
        self.rows_kept = 0
        self.by_sub_type: dict[str, int] = defaultdict(int)
        self.by_month: dict[str, int] = defaultdict(int)
        self.ai_by_sub: dict[str, list[int]] = defaultdict(lambda: [0, 0])

    def keep(self, df: pd.DataFrame, mask: pd.Series, reason: str) -> pd.DataFrame:
        mask = mask.fillna(False).astype(bool)
        n = int((~mask).sum())
        if n:
            self.dropped[reason] += n
        return df[mask]


def process_chunk(raw: pd.DataFrame, kind: str, ctx: Ctx) -> pd.DataFrame:
    a = ctx.args
    ctx.rows_read += len(raw)
    df = raw.reindex(columns=RAW_COLUMNS[kind]).copy()
    for c in df.columns:
        df[c] = clean_str(df[c])

    # 1. id
    df = ctx.keep(df, df["id"].notna(), "missing_id")

    # 2. subreddit filter + category
    sub_l = df["subreddit"].str.lower()
    df = ctx.keep(df, sub_l.isin(ctx.sub_map.keys()), "subreddit_not_in_list")
    sub_l = df["subreddit"].str.lower()
    df["category"] = sub_l.map(lambda s: ctx.sub_map[s][1])
    df["subreddit"] = sub_l.map(lambda s: ctx.sub_map[s][0])
    if df.empty:
        return df

    # 3. timestamp window
    ts = pd.to_numeric(df["created_utc"], errors="coerce")
    df = ctx.keep(df, ts.notna(), "bad_timestamp")
    ts = pd.to_numeric(df["created_utc"], errors="coerce").astype("int64")
    df = ctx.keep(df, (ts >= ctx.start_ts) & (ts < ctx.end_ts), "outside_date_range")
    if df.empty:
        return df
    ts = pd.to_numeric(df["created_utc"], errors="coerce").astype("int64")
    dt = pd.to_datetime(ts, unit="s", utc=True)
    df["created_utc"] = ts
    df["created_dt"] = dt.dt.strftime("%Y-%m-%d %H:%M:%S")
    df["month"] = dt.dt.strftime("%Y-%m")

    # 4. duplicates by (type, id)  -- within chunk and across all previous files
    df["type"] = kind
    keys = hash_rows(df["type"], df["id"])
    within = pd.Series(keys, index=df.index).duplicated().to_numpy()
    seen_before = np.fromiter((k in ctx.seen_ids for k in keys), bool, count=len(keys))
    dup = within | seen_before
    ctx.seen_ids.update(keys[~dup].tolist())
    df = ctx.keep(df, pd.Series(~dup, index=df.index), "duplicate_id")
    if df.empty:
        return df

    # 5. author + bots
    author = df["author"]
    author = author.mask(author.str.lower().isin({"[deleted]", "[removed]", "deleted"}))
    df["author"] = author
    if not a.keep_bots:
        a_low = author.str.lower()
        is_bot = a_low.isin(BOT_NAMES) | author.str.contains(BOT_REGEX, regex=True, na=False)
        df = ctx.keep(df, ~is_bot, "bot_author")
        if df.empty:
            return df

    # 5b. moderator / admin posts and stickied submissions
    df["distinguished"] = df["distinguished"].str.lower()
    df["stickied"] = to_bool(df["stickied"])
    if not a.keep_mod_posts:
        is_mod = df["distinguished"].isin(MOD_LEVELS).astype(bool)
        if kind == "submission":
            is_mod = is_mod | df["stickied"].fillna(False).astype(bool)
        df = ctx.keep(df, ~is_mod, "moderator_or_stickied")
        if df.empty:
            return df
    if kind == "submission" and a.drop_removed_posts:
        df = ctx.keep(df, df["removed_by_category"].isna(), "removed_post")
        if df.empty:
            return df

    # 6. text
    if kind == "comment":
        df["title"] = pd.NA
        body = blank_removed(df["body"])
        df["text"] = normalize_text(body)
        df["thread_id"] = df["link_id"]
        df["link_flair_text"] = pd.NA
    else:
        title = normalize_text(df["title"])
        selftext = normalize_text(blank_removed(df["selftext"]))
        df["title"] = title
        df["text"] = (title.fillna("") + "\n\n" + selftext.fillna("")).str.strip().mask(
            title.isna() & selftext.isna())
        df["thread_id"] = "t3_" + df["id"]
        df["parent_id"] = pd.NA
        df["link_flair_text"] = normalize_text(df["link_flair_text"])
        df["is_crosspost"] = df["crosspost_parent"].notna()
        df["is_self"] = to_bool(df["is_self"])
        df["url"] = df["url"].mask(df["is_self"].fillna(False).astype(bool))
    df = ctx.keep(df, df["text"].notna(), "removed_or_empty_text")
    df["n_words"] = df["text"].str.split().str.len().astype("Int64")
    df = ctx.keep(df, df["n_words"] >= a.min_words, "too_short")
    if df.empty:
        return df

    # 7. numeric / boolean fields
    for c in NUMERIC_COLS:
        if c in df.columns:
            num = pd.to_numeric(df[c], errors="coerce")
            df[c] = num if c == "upvote_ratio" else num.round().astype("Int64")
    for c in BOOL_COLS:
        if c in df.columns and c != "stickied":
            df[c] = to_bool(df[c])

    # 8. optional: repeated identical text by same author in same subreddit
    if a.dedupe_text:
        has_author = df["author"].notna()
        h = hash_rows(df["subreddit"], df["author"], df["text"])
        within = pd.Series(h, index=df.index).duplicated().to_numpy()
        seen_before = np.fromiter((x in ctx.seen_text for x in h), bool, count=len(h))
        dup = (within | seen_before) & has_author.to_numpy()
        ctx.seen_text.update(h[has_author.to_numpy() & ~dup].tolist())
        df = ctx.keep(df, pd.Series(~dup, index=df.index), "duplicate_text")
        if df.empty:
            return df

    # 9. optional: AI-mention flag + optional drop
    df["mentions_ai"] = df["text"].str.contains(AI_REGEX, case=False, regex=True, na=False)
    if ctx.require_ai:
        drop = df["category"].isin(ctx.require_ai) & ~df["mentions_ai"]
        df = ctx.keep(df, ~drop, "no_ai_mention_in_required_category")

    # 10. finalise
    df = df.reindex(columns=OUTPUT_COLUMNS)
    ctx.rows_kept += len(df)
    for (sub, typ), n in df.groupby(["subreddit", "type"]).size().items():
        ctx.by_sub_type[f"{sub}|{typ}"] += int(n)
    for m, n in df.groupby("month").size().items():
        ctx.by_month[m] += int(n)
    for sub, g in df.groupby("subreddit")["mentions_ai"]:
        ctx.ai_by_sub[sub][0] += int(g.sum())
        ctx.ai_by_sub[sub][1] += int(len(g))
    return df


def thread_filter_pass(out_path: Path, categories: set[str], chunksize: int, ctx: Ctx) -> None:
    log.info("Thread-level pass A: finding AI-related threads ...")
    parts = []
    for ch in pd.read_csv(out_path, dtype=str, keep_default_na=False,
                          usecols=["thread_id", "mentions_ai"], chunksize=chunksize):
        sel = ch[(ch["mentions_ai"] == "True") & (ch["thread_id"] != "")]
        if len(sel):
            parts.append(pd.util.hash_pandas_object(sel["thread_id"], index=False).to_numpy())
    ai_threads = np.unique(np.concatenate(parts)) if parts else np.array([], dtype=np.uint64)
    log.info("  %d threads contain at least one AI mention", len(ai_threads))

    log.info("Thread-level pass B: keeping every row of AI threads ...")
    tmp = out_path.with_suffix(".tmp")
    if tmp.exists():
        tmp.unlink()
    by_sub_type, by_month = defaultdict(int), defaultdict(int)
    ai_by_sub = defaultdict(lambda: [0, 0])
    kept, first = 0, True
    for ch in pd.read_csv(out_path, dtype=str, keep_default_na=False, chunksize=chunksize):
        h = pd.util.hash_pandas_object(ch["thread_id"], index=False).to_numpy()
        own = (ch["mentions_ai"] == "True").to_numpy()
        has_tid = (ch["thread_id"] != "").to_numpy()
        flag = np.where(has_tid, np.isin(h, ai_threads), own)
        ch["thread_mentions_ai"] = flag
        drop = ch["category"].isin(categories).to_numpy() & ~flag
        ctx.dropped["thread_without_ai_mention"] += int(drop.sum())
        ch = ch[~drop]
        if ch.empty:
            continue
        ch.to_csv(tmp, mode="a", header=first, index=False)
        first = False
        kept += len(ch)
        for (sub, typ), n in ch.groupby(["subreddit", "type"]).size().items():
            by_sub_type[f"{sub}|{typ}"] += int(n)
        for m, n in ch.groupby("month").size().items():
            by_month[m] += int(n)
        for sub, g in ch.groupby("subreddit")["mentions_ai"]:
            ai_by_sub[sub][0] += int((g == "True").sum())
            ai_by_sub[sub][1] += int(len(g))
    if first: 
        tmp.unlink(missing_ok=True)
        ctx.rows_kept = 0
        return
    tmp.replace(out_path)
    ctx.rows_kept, ctx.by_sub_type, ctx.by_month, ctx.ai_by_sub = kept, by_sub_type, by_month, ai_by_sub


# MAIN

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("inputs", nargs="+", help="Folders (searched recursively) or RC_/RS_ CSV files")
    p.add_argument("-o", "--output", default="data/processed/reddit_ai_psychosis.csv")
    p.add_argument("--start", default="2025-05-01", help="First day kept, UTC (YYYY-MM-DD)")
    p.add_argument("--end", default="2026-08-31", help="Last day kept, UTC, inclusive")
    p.add_argument("--subreddits", type=lambda s: {x.strip().lower() for x in s.split(",")},
                   default=None, help="Comma list to restrict to a subset of SUBREDDITS")
    p.add_argument("--require-ai-in", type=lambda s: [x.strip() for x in s.split(",")],
                   default=None, metavar="CATEGORIES",
                   help="Drop rows WITHOUT an AI keyword in these categories "
                        "(ai_partners, ai_sentience, mental_health, general_ai)")
    p.add_argument("--thread-ai-in", type=lambda s: [x.strip() for x in s.split(",")],
                   default=None, metavar="CATEGORIES",
                   help="THREAD-level AI filter: in these categories keep a thread (post + ALL "
                        "its comments) only if the post or any comment mentions AI. "
                        "Cannot be combined with --require-ai-in.")
    p.add_argument("--thread-flags", action="store_true",
                   help="Add the `thread_mentions_ai` column WITHOUT dropping anything "
                        "(keeps the full baseline; filter later in pandas).")
    p.add_argument("--min-words", type=int, default=1, help="Drop texts shorter than N words")
    p.add_argument("--keep-bots", action="store_true", help="Do not drop bot authors")
    p.add_argument("--dedupe-text", action="store_true",
                   help="Drop repeated identical text by same author in same subreddit")
    p.add_argument("--keep-mod-posts", action="store_true",
                   help="Keep moderator/admin-distinguished items and stickied submissions")
    p.add_argument("--drop-removed-posts", action="store_true",
                   help="Drop submissions with removed_by_category set (default: keep the title)")
    p.add_argument("--chunksize", type=int, default=200_000)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s",
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(out_path.with_suffix(".log"), mode="w")])

    if args.thread_ai_in and args.require_ai_in:
        log.error("--thread-ai-in and --require-ai-in are mutually exclusive")
        return 1
    if out_path.exists() and not args.overwrite:
        log.error("%s exists. use --overwrite to replace it.", out_path)
        return 1
    if out_path.exists():
        out_path.unlink()

    files = discover_files(args.inputs)
    if not files:
        log.error("no RC_YYYY-MM.csv / RS_YYYY-MM.csv files found in %s", args.inputs)
        return 1
    log.info("found %d files; months %s .. %s", len(files), files[0][0], files[-1][0])

    ctx = Ctx(args)
    files_report, first = [], True
    for month, kind, path in files:
        rows_before = ctx.rows_kept
        try:
            wanted = set(RAW_COLUMNS[kind])
            reader = pd.read_csv(path, dtype=str, keep_default_na=False,
                                 usecols=lambda c: c in wanted,
                                 chunksize=args.chunksize, encoding="utf-8",
                                 encoding_errors="replace", on_bad_lines="skip")
            for raw in reader:
                out = process_chunk(raw, kind, ctx)
                if out is not None and not out.empty:
                    out.to_csv(out_path, mode="a", header=first, index=False)
                    first = False
        except pd.errors.EmptyDataError:
            log.warning("empty file skipped: %s", path)
        added = ctx.rows_kept - rows_before
        files_report.append({"file": str(path), "kind": kind, "kept": added})
        log.info("%s %-10s -> +%d rows (total %d)", month, kind, added, ctx.rows_kept)

    if first:
        log.error("nothing survived filtering")
        return 1

    if args.thread_ai_in or args.thread_flags:
        thread_filter_pass(out_path, set(args.thread_ai_in or []), args.chunksize, ctx)
        if ctx.rows_kept == 0:
            log.error("thread-level filter removed everything")
            return 1

    size_mb = out_path.stat().st_size / 1024 / 1024
    report = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "args": {k: (sorted(v) if isinstance(v, set) else v) for k, v in vars(args).items()},
        "output": str(out_path), "output_size_mb": round(size_mb, 2),
        "rows_read": ctx.rows_read, "rows_kept": ctx.rows_kept,
        "dropped_by_reason": dict(ctx.dropped),
        "kept_by_subreddit_type": dict(sorted(ctx.by_sub_type.items())),
        "kept_by_month": dict(sorted(ctx.by_month.items())),
        "mentions_ai_share_by_subreddit": {
            s: round(v[0] / v[1], 4) for s, v in sorted(ctx.ai_by_sub.items())},
        "files": files_report,
    }
    report_path = out_path.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    log.info("=" * 60)
    log.info("read %d rows -> kept %d  |  %.1f MB", ctx.rows_read, ctx.rows_kept, size_mb)
    for reason, n in sorted(ctx.dropped.items(), key=lambda x: -x[1]):
        log.info("  dropped %-38s %d", reason, n)
    log.info("output: %s | Report: %s", out_path, report_path)
    if size_mb < 1024:
        log.warning("output is %.0f MB.", size_mb)
    return 0


if __name__ == "__main__":
    sys.exit(main())