"""
Collect and prepare the Project Gutenberg corpus.

Gutenberg is the project's only long-form source: ~75,000 public-domain English
books of edited literary prose, against Bluesky posts, Reddit threads and
TinyStories' deliberately tiny vocabulary. Measured over a 56k-book download, the
full English subset is ~24 GB of plain text and ~5 billion tokens, which forces two
things the other corpora never needed. (The 1-2 TB figure quoted for Project
Gutenberg is the complete mirror -- every format and language, images, audio and
superseded revisions. This collector fetches one plain-text file per English book.)

**Everything streams.** The other collectors build one in-memory DataFrame and
the other postprocessors call ``pd.read_parquet`` on the whole file. Neither
survives this corpus, so collection and postprocessing both read and write in
batches through ``pyarrow.parquet``, and deduplication keeps digests rather than
rows.

**Downloads are cached and polite.** ``gutenberg.org`` blocks bulk crawling and
directs heavy users to its mirrors, so books come from a configurable mirror with
a request delay, and only the catalog is fetched from the main site. Raw books
land in ``data/gutenberg_raw/`` and are never re-fetched, which makes a run that
takes about a day interruptible and resumable:

    t2-collect --dataset gutenberg --max_books 20     # try the mirror first
    t2-collect --dataset gutenberg                    # then the whole subset
    t2-tokenizer --dataset_type gutenberg
    t2-postprocess gutenberg --max_tokens 512

Note that stripping the Project Gutenberg header, which training requires, means
the resulting corpus may not be redistributed under the Gutenberg trademark.
"""

import gzip
import hashlib
import io
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests
from tqdm import tqdm

from transformer2.data.chunking import chunk_segments
from transformer2.paths import data_dir, ensure
from transformer2.tokenization.bpe import create_tokenizer

CATALOG_URL = "https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv.gz"
# xmission is an official Gutenberg mirror that serves a valid TLS certificate;
# aleph.gutenberg.org carries the same tree but only over plain HTTP. Override
# with --mirror if this one is slow or unreachable.
DEFAULT_MIRROR = "https://mirrors.xmission.com/gutenberg"
USER_AGENT = "Transformer2/0.1 (research corpus collection; one request per second)"

RAW_DIR_NAME = "gutenberg_raw"
CATALOG_NAME = "catalog.csv"
INPUT_NAME = "gutenberg_books.parquet"
OUTPUT_NAME = "gutenberg_text.parquet"

# Books shorter than this after boilerplate removal are almost always stubs:
# an index page, a placeholder, or a file whose markers we failed to match.
MIN_BOOK_CHARS = 2000

# Rows buffered before a parquet row group is flushed. The corpus never sits in
# memory in full; this only bounds the write buffer.
FLUSH_EVERY = 500

_START_RE = re.compile(
    r'\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*',
    re.IGNORECASE | re.DOTALL)
_END_RE = re.compile(
    r'\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*',
    re.IGNORECASE | re.DOTALL)
# Files predating the *** markers close with a bare sentence instead
_END_SENTINEL_RE = re.compile(
    r'^End of (?:the )?(?:Project Gutenberg|Project Gutenberg\'s).*$',
    re.IGNORECASE | re.MULTILINE)
_PRODUCED_BY_RE = re.compile(
    r'\A\s*(?:Produced by|Transcribed by|E-text prepared by)\b.*?(?:\n\s*\n)',
    re.IGNORECASE | re.DOTALL)

_TRAILING_SPACE_RE = re.compile(r'[ \t]+')
_BLANK_LINES_RE = re.compile(r'\n{3,}')


def raw_dir():
    """Directory holding the cached raw book downloads."""
    return data_dir() / RAW_DIR_NAME


def mirror_paths(book_id, mirror=DEFAULT_MIRROR):
    """
    Candidate URLs for one book on a Gutenberg mirror.

    Mirrors disperse books across a directory tree built from the digits of the
    id: book 12345 lives under ``1/2/3/4/12345/``, and a single-digit id under
    ``0/<id>/``. Each book is published in several encodings, so the caller tries
    these in order: ``-0`` is UTF-8, the bare name is ASCII, ``-8`` is Latin-1.

    Args:
        book_id (int or str): Gutenberg text number
        mirror (str): Base URL of the mirror

    Returns:
        list of str: Candidate URLs, most preferred first
    """
    book_id = str(book_id)
    # The last digit names the book's own directory, so only the preceding
    # digits form the dispersed path; single-digit ids have none and use "0".
    prefix = "/".join(book_id[:-1]) if len(book_id) > 1 else "0"
    base = f"{mirror.rstrip('/')}/{prefix}/{book_id}"
    return [f"{base}/{book_id}{suffix}.txt" for suffix in ("-0", "", "-8")]


def _decode(content):
    """Decode a downloaded book, falling back from UTF-8 to cp1252."""
    for encoding in ("utf-8", "cp1252", "latin-1"):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    return content.decode("utf-8", errors="replace")


def _normalise_catalog(books):
    """Coerce the catalog columns to the types the parquet writer expects."""
    return pd.DataFrame({
        "book_id": books["book_id"].astype(int),
        # Catalog titles carry embedded newlines from the CSV export, and an
        # absent author reads back as NaN rather than a string.
        "title": books["title"].fillna("").astype(str).str.replace(
            r'\s+', ' ', regex=True).str.strip(),
        "author": books["author"].fillna("").astype(str).str.replace(
            r'\s+', ' ', regex=True).str.strip(),
    }).reset_index(drop=True)


def fetch_catalog(url=CATALOG_URL, language="en", cache_path=None, session=None):
    """
    Download the Gutenberg catalog and keep the books worth collecting.

    One request for a ~10 MB gzipped CSV, cached afterwards so repeated runs and
    resumes never fetch it again. This is the only request made to gutenberg.org
    itself; the books come from a mirror.

    Args:
        url (str): Catalog URL
        language (str): Keep only books in this language code
        cache_path (str or pathlib.Path, optional): Where to cache the filtered
            catalog. Defaults to data/gutenberg_raw/catalog.csv.
        session (requests.Session, optional): Session to fetch with

    Returns:
        pandas.DataFrame: Columns book_id, title, author
    """
    cache_path = str(cache_path or raw_dir() / CATALOG_NAME)
    if os.path.exists(cache_path):
        print(f"Using the cached catalog at {cache_path}")
        # A CSV round-trip reads an empty title back as NaN, so re-normalise:
        # a float reaching the parquet writer fails the whole run.
        return _normalise_catalog(pd.read_csv(cache_path))

    print(f"Downloading the Project Gutenberg catalog from {url}")
    session = session or requests.Session()
    response = session.get(url, headers={"User-Agent": USER_AGENT}, timeout=120)
    response.raise_for_status()

    raw = response.content
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    catalog = pd.read_csv(io.BytesIO(raw))

    keep = catalog[(catalog["Type"] == "Text") & (catalog["Language"] == language)]
    books = _normalise_catalog(pd.DataFrame({
        "book_id": keep["Text#"],
        "title": keep["Title"],
        "author": keep["Authors"],
    }))

    ensure(cache_path)
    books.to_csv(cache_path, index=False)
    print(f"{len(books):,} {language} books in the catalog ({len(catalog):,} entries total)")
    return books


def download_book(book_id, mirror=DEFAULT_MIRROR, session=None, delay=1.0,
                  retries=3, cache_dir=None):
    """
    Fetch one book, reusing the on-disk cache when it is already there.

    The cache is what makes a full collection resumable: a run interrupted after
    40,000 books re-reads those from disk and issues no request for them.

    Args:
        book_id (int or str): Gutenberg text number
        mirror (str): Mirror base URL
        session (requests.Session, optional): Session to fetch with
        delay (float): Seconds to wait after each request
        retries (int): Attempts per URL for rate limits and server errors
        cache_dir (str or pathlib.Path, optional): Raw cache directory

    Returns:
        tuple: (text, from_cache). text is None when every candidate URL 404s.
    """
    cache_dir = cache_dir or raw_dir()
    cached = os.path.join(str(cache_dir), f"{book_id}.txt")
    if os.path.exists(cached):
        with open(cached, encoding="utf-8") as handle:
            return handle.read(), True

    session = session or requests.Session()
    for url in mirror_paths(book_id, mirror):
        for attempt in range(retries):
            try:
                response = session.get(url, headers={"User-Agent": USER_AGENT},
                                       timeout=60)
            except requests.RequestException as error:
                print(f"  {book_id}: {error}")
                time.sleep(delay * (2 ** attempt))
                continue
            finally:
                time.sleep(delay)

            if response.status_code == 404:
                break  # This encoding does not exist; try the next candidate
            if response.status_code in (429, 500, 502, 503, 504):
                # Back off rather than hammering a mirror that is asking us not to
                wait = response.headers.get("Retry-After")
                time.sleep(float(wait) if wait and wait.isdigit()
                           else delay * (2 ** attempt))
                continue

            response.raise_for_status()
            text = _decode(response.content)
            ensure(cached)
            with open(cached, "w", encoding="utf-8") as handle:
                handle.write(text)
            return text, False

    return None, False


def strip_boilerplate(text):
    """
    Remove the Project Gutenberg header, footer and license.

    The markers' wording drifted over decades, so both the modern ``*** START OF
    THE PROJECT GUTENBERG EBOOK ... ***`` banners and the older bare closing
    sentence are matched. A file with no recognisable start marker returns an
    empty string rather than the license text: the caller counts those as skipped,
    because training on the license would teach the model to reproduce it.

    Args:
        text (str): Raw book file

    Returns:
        str: The book's own text, or "" if the markers were not found
    """
    if not isinstance(text, str):
        return ""

    start = _START_RE.search(text)
    if start is None:
        return ""
    body = text[start.end():]

    end = _END_RE.search(body) or _END_SENTINEL_RE.search(body)
    if end is not None:
        body = body[:end.start()]

    return _PRODUCED_BY_RE.sub("", body).strip()


def clean_book(text):
    """
    Normalise a book's whitespace.

    Gutenberg text is hard-wrapped at about 70 columns, so a single newline is a
    line break inside a paragraph rather than a break between paragraphs. It has
    to be joined into a space: chunking splits on blank lines, and leaving the
    wrapping in place would make every book one unsplittable paragraph.

    Args:
        text (str): Book text with the boilerplate already removed

    Returns:
        str: Normalised text with paragraphs separated by blank lines
    """
    if not isinstance(text, str):
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _BLANK_LINES_RE.sub("\n\n", text)

    paragraphs = []
    for block in text.split("\n\n"):
        joined = " ".join(line.strip() for line in block.split("\n") if line.strip())
        joined = _TRAILING_SPACE_RE.sub(" ", joined).strip()
        if joined:
            paragraphs.append(joined)
    return "\n\n".join(paragraphs)


def _book_schema():
    return pa.schema([
        ("text", pa.string()),
        ("book_id", pa.int64()),
        ("title", pa.string()),
        ("author", pa.string()),
    ])


def _pid_is_running(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _acquire_lock(output_path):
    """
    Take an exclusive lock on a corpus path.

    Two collections writing one parquet interleave their row groups and leave a file
    whose footer is valid but whose middle is unreadable -- the failure only surfaces
    later, when a reader reaches the damaged band. Fail loudly at the start instead.

    Args:
        output_path (str): Corpus being written

    Returns:
        pathlib.Path: The lock file, to pass to _release_lock

    Raises:
        RuntimeError: If a live process already holds the lock
    """
    lock = Path(f"{output_path}.lock")
    while True:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                holder = int(lock.read_text().strip())
            except (ValueError, OSError):
                holder = None
            if holder is not None and holder != os.getpid() and _pid_is_running(holder):
                raise RuntimeError(
                    f"Another collection (pid {holder}) is already writing "
                    f"{output_path}.\n"
                    f"Wait for it to finish, or remove {lock} if that process is gone."
                )
            # The holder died without cleaning up; take the lock over.
            lock.unlink(missing_ok=True)
            continue
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        return lock


def _release_lock(lock):
    Path(lock).unlink(missing_ok=True)


def collect_gutenberg(output_path=None, mirror=DEFAULT_MIRROR, max_books=None,
                      delay=1.0, language="en", catalog_url=CATALOG_URL,
                      session=None, cache_dir=None, min_chars=MIN_BOOK_CHARS):
    """
    Download the Project Gutenberg corpus and write it as parquet.

    Rows are flushed in row groups as they are collected, so the corpus never sits
    in memory in full. Raw downloads are cached under data/gutenberg_raw/, which
    makes an interrupted run resumable at no cost.

    Args:
        output_path (str, optional): Destination parquet
        mirror (str): Mirror base URL to download books from
        max_books (int, optional): Stop after this many books. All of them if None.
        delay (float): Seconds between requests
        language (str): Catalog language filter
        catalog_url (str): Catalog URL
        session (requests.Session, optional): Session to fetch with
        cache_dir (str or pathlib.Path, optional): Raw cache directory
        min_chars (int): Discard books shorter than this after cleaning

    Returns:
        dict: Counts of written, cached, skipped and failed books
    """
    output_path = str(output_path or data_dir() / INPUT_NAME)
    cache_dir = cache_dir or raw_dir()
    session = session or requests.Session()

    catalog = fetch_catalog(catalog_url, language=language, session=session)
    if max_books is not None:
        catalog = catalog.head(max_books)

    if max_books is None:
        print(f"Collecting {len(catalog):,} books from {mirror} at {delay}s per request.")
        print("  This takes many hours. Interrupt it freely -- downloads are cached")
        print(f"  under {cache_dir} and a re-run continues where it stopped.")

    ensure(output_path)
    counts = {"written": 0, "cached": 0, "skipped": 0, "failed": 0}
    buffer, writer = [], None

    # Build beside the target and rename at the end. A reader therefore only ever
    # sees a complete corpus, and a run that dies partway leaves the previous one
    # intact rather than a half-written file that parses until it does not.
    lock = _acquire_lock(output_path)
    temp_path = f"{output_path}.partial-{os.getpid()}"
    finished = False

    def flush():
        nonlocal buffer, writer
        if not buffer:
            return
        table = pa.Table.from_pylist(buffer, schema=_book_schema())
        if writer is None:
            writer = pq.ParquetWriter(temp_path, _book_schema())
        writer.write_table(table)
        buffer = []

    try:
        for row in tqdm(catalog.itertuples(index=False), total=len(catalog),
                        desc="Collecting", unit="book"):
            raw, from_cache = download_book(row.book_id, mirror=mirror,
                                            session=session, delay=delay,
                                            cache_dir=cache_dir)
            if raw is None:
                counts["failed"] += 1
                continue
            if from_cache:
                counts["cached"] += 1

            text = clean_book(strip_boilerplate(raw))
            if len(text) < min_chars:
                counts["skipped"] += 1
                continue

            buffer.append({"text": text, "book_id": int(row.book_id),
                           "title": row.title, "author": row.author})
            counts["written"] += 1
            if len(buffer) >= FLUSH_EVERY:
                flush()
        finished = True
    except KeyboardInterrupt:
        # Interrupting a day-long run is expected; keep what was collected. The raw
        # cache means a re-run rebuilds the rest without re-downloading anything.
        print("\nInterrupted -- keeping the books collected so far.")
        finished = True
        raise
    finally:
        try:
            flush()
            if writer is not None:
                writer.close()
            if writer is None and finished:
                # An empty corpus still needs a readable file, or postprocessing
                # reports a missing download rather than an empty one.
                pq.write_table(pa.Table.from_pylist([], schema=_book_schema()),
                               temp_path)
            if finished:
                os.replace(temp_path, output_path)
            elif os.path.exists(temp_path):
                # Left in place deliberately: the run failed, so the previous corpus
                # stays authoritative and the partial file is available to inspect.
                print(f"Collection failed; partial output left at {temp_path}")
        finally:
            _release_lock(lock)

    print(f"\nWrote {counts['written']:,} books to {output_path}")
    print(f"  already cached: {counts['cached']:,}")
    print(f"  skipped (too short or no Gutenberg markers): {counts['skipped']:,}")
    print(f"  unavailable on the mirror: {counts['failed']:,}")
    return counts


def postprocess_gutenberg(input_path=None, output_path=None, max_tokens=512,
                          min_tokens=32, tokenizer=None, drop_duplicates=True,
                          batch_size=200):
    """
    Chunk the collected books into training rows.

    Each book is chunked on its own, so a chunk never runs from the end of one
    work into the start of an unrelated one -- the same rule TinyStories applies
    per story. Books are read and rows written in batches, and duplicates are
    tracked as digests rather than rows, because the full corpus does not fit in
    memory.

    Args:
        input_path (str, optional): Parquet written by collect_gutenberg
        output_path (str, optional): Destination parquet with a single text column
        max_tokens (int): Maximum tokens per row; match the training --max_length
        min_tokens (int): Discard rows shorter than this
        tokenizer: Tokenizer wrapper used to measure lengths. Created if None.
        drop_duplicates (bool): Remove identical chunks
        batch_size (int): Books read per parquet batch

    Returns:
        dict: Counts and token statistics for the written corpus

    Raises:
        FileNotFoundError: If input_path does not exist
    """
    input_path = str(input_path or data_dir() / INPUT_NAME)
    output_path = str(output_path or data_dir() / OUTPUT_NAME)

    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"{input_path} not found. Collect it first:\n"
            f"    t2-collect --dataset gutenberg"
        )

    if tokenizer is None:
        tokenizer = create_tokenizer("gutenberg")

    def lengths_fn(texts):
        texts = list(texts)
        if not texts:
            return []
        return [len(e['input_ids']) for e in tokenizer.encode_batch(texts)]

    reader = pq.ParquetFile(input_path)
    total_books = reader.metadata.num_rows
    print(f"Loaded {total_books:,} books from {input_path}")

    schema = pa.schema([("text", pa.string())])
    ensure(output_path)
    writer = pq.ParquetWriter(output_path, schema)

    seen = set()
    buffer = []
    rows = duplicates = 0
    token_total = 0
    token_min, token_max = None, None
    sample = []

    def flush():
        nonlocal buffer
        if not buffer:
            return
        writer.write_table(pa.Table.from_pylist(
            [{"text": t} for t in buffer], schema=schema))
        buffer = []

    try:
        for batch in tqdm(reader.iter_batches(batch_size=batch_size, columns=["text"]),
                          total=max(1, -(-total_books // batch_size)),
                          desc="Chunking", unit="batch"):
            for book in batch.column("text").to_pylist():
                book = clean_book(book)
                if not book:
                    continue
                # One book per call: this is what stops a chunk spanning two works
                chunks = chunk_segments([book], lengths_fn, max_tokens, min_tokens)
                if not chunks:
                    continue

                for chunk, length in zip(chunks, lengths_fn(chunks)):
                    if drop_duplicates:
                        digest = hashlib.blake2b(chunk.encode("utf-8"),
                                                 digest_size=8).digest()
                        if digest in seen:
                            duplicates += 1
                            continue
                        seen.add(digest)

                    buffer.append(chunk)
                    rows += 1
                    token_total += length
                    token_min = length if token_min is None else min(token_min, length)
                    token_max = length if token_max is None else max(token_max, length)
                    if len(sample) < 100000:
                        sample.append(length)

                if len(buffer) >= FLUSH_EVERY:
                    flush()
    finally:
        flush()
        writer.close()

    stats = {
        "rows": rows,
        "duplicates_removed": duplicates,
        "total_tokens": token_total,
        "mean_tokens": (token_total / rows) if rows else 0,
        "min_tokens": token_min or 0,
        "max_tokens": token_max or 0,
    }

    print(f"\nWrote {rows:,} rows to {output_path}")
    if drop_duplicates:
        print(f"  removed {duplicates:,} duplicates")
    if rows:
        print(f"  tokens per row: mean {stats['mean_tokens']:.0f}, "
              f"min {stats['min_tokens']}, max {stats['max_tokens']}, "
              f"p95 {int(np.percentile(sample, 95))}")
        print(f"  fill at max_length={max_tokens}: "
              f"{100 * stats['mean_tokens'] / max_tokens:.0f}%")
        print(f"  total tokens: {token_total:,}")
    return stats
