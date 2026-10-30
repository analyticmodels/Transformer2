"""
Tests for Project Gutenberg collection and postprocessing.

Gutenberg is the only book-length corpus here, so the work is stripping the
Project Gutenberg boilerplate, undoing the hard wrapping, and chunking each book
on its own. The collector is exercised against a fake session; nothing here
reaches the network.
"""

import gzip
import os

import pandas as pd
import pytest

from transformer2.data.gutenberg import (
    clean_book,
    collect_gutenberg,
    download_book,
    fetch_catalog,
    mirror_paths,
    postprocess_gutenberg,
    strip_boilerplate,
)

MIRROR = "https://mirror.example"

BOOK = """The Project Gutenberg eBook of Some Book

*** START OF THE PROJECT GUTENBERG EBOOK SOME BOOK ***

Produced by A Volunteer and the Online
Distributed Proofreading Team.

Call me Ishmael. Some years ago, never
mind how long precisely, I thought I
would sail about a little.

It is a way I have of driving off the
spleen.

*** END OF THE PROJECT GUTENBERG EBOOK SOME BOOK ***

START: FULL LICENSE. This must never reach the corpus.
"""


class WordTokenizer:
    """Whitespace tokenizer with the wrapper's encode_batch contract."""

    def encode_batch(self, texts):
        return [{'input_ids': t.split(), 'attention_mask': [1] * len(t.split())}
                for t in texts]

    def encode(self, text):
        return {'input_ids': text.split(), 'attention_mask': [1] * len(text.split())}


class FakeResponse:
    def __init__(self, body=b"", status_code=200):
        self.content = body
        self.status_code = status_code
        self.headers = {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise AssertionError(f"unexpected raise_for_status at {self.status_code}")


class FakeSession:
    """Serves a catalog and a fixed set of book bodies, recording every URL."""

    def __init__(self, bodies, catalog=None):
        self.bodies = bodies
        self.catalog = catalog
        self.urls = []

    def get(self, url, **kwargs):
        self.urls.append(url)
        if url.endswith('.csv.gz'):
            return FakeResponse(gzip.compress(self.catalog.encode()))
        if url in self.bodies:
            return FakeResponse(self.bodies[url].encode())
        return FakeResponse(b"", status_code=404)


CATALOG = (
    "Text#,Type,Issued,Title,Language,Authors,Subjects,LoCC,Bookshelves\n"
    "1,Text,1971-12-01,Book One,en,Author One,,,\n"
    "2,Text,1972-01-01,Book Two,en,Author Two,,,\n"
    "3,Text,1972-01-01,Livre Trois,fr,Auteur,,,\n"
    "4,Sound,1972-01-01,An Audiobook,en,Author,,,\n"
)


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    """Point every project path at a temporary directory."""
    monkeypatch.setenv("TRANSFORMER2_HOME", str(tmp_path))
    return tmp_path


class TestMirrorPaths:
    def test_disperses_by_digit(self):
        urls = mirror_paths(12345, MIRROR)
        assert urls[0] == f"{MIRROR}/1/2/3/4/12345/12345-0.txt"

    def test_single_digit_id_uses_zero_prefix(self):
        assert mirror_paths(7, MIRROR)[0] == f"{MIRROR}/0/7/7-0.txt"

    def test_encodings_are_tried_in_preference_order(self):
        urls = mirror_paths(42, MIRROR)
        assert [u.rsplit('/', 1)[1] for u in urls] == ['42-0.txt', '42.txt', '42-8.txt']

    def test_trailing_slash_on_mirror_is_tolerated(self):
        assert '//' not in mirror_paths(42, MIRROR + '/')[0].replace('https://', '')


class TestStripBoilerplate:
    def test_keeps_only_the_body(self):
        body = strip_boilerplate(BOOK)
        assert 'Call me Ishmael' in body
        assert 'FULL LICENSE' not in body
        assert '***' not in body

    def test_drops_the_transcriber_block(self):
        assert 'Proofreading Team' not in strip_boilerplate(BOOK)

    def test_accepts_the_older_this_wording(self):
        text = ("*** START OF THIS PROJECT GUTENBERG EBOOK X ***\nBody here.\n"
                "*** END OF THIS PROJECT GUTENBERG EBOOK X ***\nLicense.")
        assert strip_boilerplate(text).strip() == 'Body here.'

    def test_accepts_the_bare_closing_sentinel(self):
        text = ("*** START OF THE PROJECT GUTENBERG EBOOK X ***\nBody here.\n"
                "End of the Project Gutenberg EBook of X, by Y\nLicense.")
        assert 'License' not in strip_boilerplate(text)

    def test_missing_start_marker_yields_nothing(self):
        assert strip_boilerplate("A file with no markers at all.") == ""

    def test_non_string_is_safe(self):
        assert strip_boilerplate(None) == ""


class TestCleanBook:
    def test_joins_hard_wrapped_lines(self):
        assert clean_book("Call me\nIshmael.") == "Call me Ishmael."

    def test_blank_lines_remain_paragraph_breaks(self):
        assert clean_book("One\nline.\n\nTwo\nline.") == "One line.\n\nTwo line."

    def test_collapses_runs_of_blank_lines(self):
        assert '\n\n\n' not in clean_book("a\n\n\n\n\nb")

    def test_non_string_is_safe(self):
        assert clean_book(None) == ""
        assert clean_book(float('nan')) == ""


class TestPostprocess:
    def _run(self, books, tmp_path, **kwargs):
        src = tmp_path / 'gutenberg_books.parquet'
        out = tmp_path / 'out.parquet'
        pd.DataFrame(books).to_parquet(src, index=False)
        kwargs.setdefault('max_tokens', 20)
        kwargs.setdefault('min_tokens', 3)
        postprocess_gutenberg(str(src), str(out), tokenizer=WordTokenizer(), **kwargs)
        return pd.read_parquet(out)

    def _books(self, texts):
        return {'text': texts, 'book_id': list(range(len(texts))),
                'title': ['t'] * len(texts), 'author': ['a'] * len(texts)}

    def test_writes_single_text_column(self, tmp_path):
        out = self._run(self._books([" ".join(f"w{i}" for i in range(60))]), tmp_path)
        assert out.columns.tolist() == ['text']

    def test_no_row_exceeds_max_tokens(self, tmp_path):
        out = self._run(self._books([" ".join(f"w{i}" for i in range(300))]), tmp_path)
        assert out['text'].str.split().str.len().max() <= 20

    def test_long_books_are_chunked_not_truncated(self, tmp_path):
        book = " ".join(f"w{i}" for i in range(200))
        out = self._run(self._books([book]), tmp_path)
        assert len(out) > 1
        # The tail of the book survives somewhere in the output
        assert any('w199' in row for row in out['text'])

    def test_chunks_never_span_two_books(self, tmp_path):
        first = "\n\n".join(f"alpha{i} " * 5 for i in range(4))
        second = "\n\n".join(f"beta{i} " * 5 for i in range(4))
        out = self._run(self._books([first, second]), tmp_path)
        for row in out['text']:
            assert not ('alpha' in row and 'beta' in row)

    def test_short_chunks_are_dropped(self, tmp_path):
        out = self._run(self._books(["one two"]), tmp_path, min_tokens=5)
        assert len(out) == 0

    def test_duplicates_are_removed(self, tmp_path):
        book = " ".join(f"w{i}" for i in range(30))
        out = self._run(self._books([book, book]), tmp_path)
        assert len(out) == len(set(out['text']))

    def test_duplicates_kept_when_requested(self, tmp_path):
        book = " ".join(f"w{i}" for i in range(30))
        kept = self._run(self._books([book, book]), tmp_path, drop_duplicates=False)
        deduped = self._run(self._books([book, book]), tmp_path)
        assert len(kept) > len(deduped)

    def test_hard_wrapping_is_undone(self, tmp_path):
        out = self._run(self._books(["Call me\nIshmael and\nthen some more words here"]),
                        tmp_path, min_tokens=1)
        assert "Call me Ishmael" in out['text'].iloc[0]

    def test_missing_input_raises_with_guidance(self, tmp_path):
        with pytest.raises(FileNotFoundError, match=r"t2-collect --dataset gutenberg"):
            postprocess_gutenberg(str(tmp_path / 'nope.parquet'),
                                  str(tmp_path / 'out.parquet'),
                                  tokenizer=WordTokenizer(), max_tokens=20)


class TestCatalog:
    def test_keeps_only_english_texts(self, data_root):
        session = FakeSession({}, catalog=CATALOG)
        books = fetch_catalog(language='en', session=session)
        assert books['book_id'].tolist() == [1, 2]
        assert books['title'].tolist() == ['Book One', 'Book Two']

    def test_absent_author_survives_the_cache_round_trip(self, data_root):
        # An empty field reads back from the CSV cache as NaN, and a float
        # reaching the parquet writer aborts the whole collection.
        catalog = CATALOG + "5,Text,1972-01-01,Anonymous Work,en,,,,\n"
        session = FakeSession({}, catalog=catalog)
        fetch_catalog(language='en', session=session)
        again = fetch_catalog(language='en', session=session)
        assert again['author'].tolist() == ['Author One', 'Author Two', '']
        assert all(isinstance(a, str) for a in again['author'])

    def test_whitespace_in_titles_is_collapsed(self, data_root):
        catalog = CATALOG.replace(',Book One,', ',"Book\r\nOne",')
        books = fetch_catalog(language='en', session=FakeSession({}, catalog=catalog))
        assert books['title'].iloc[0] == 'Book One'

    def test_is_cached_after_the_first_fetch(self, data_root):
        session = FakeSession({}, catalog=CATALOG)
        fetch_catalog(language='en', session=session)
        fetch_catalog(language='en', session=session)
        assert len(session.urls) == 1


class TestDownload:
    def test_falls_back_through_encodings(self, data_root):
        url = f"{MIRROR}/0/1/1.txt"
        session = FakeSession({url: "body"})
        text, from_cache = download_book(1, mirror=MIRROR, session=session, delay=0)
        assert (text, from_cache) == ("body", False)
        assert session.urls[0].endswith('1-0.txt')

    def test_cached_book_issues_no_request(self, data_root):
        url = f"{MIRROR}/0/1/1-0.txt"
        session = FakeSession({url: "body"})
        download_book(1, mirror=MIRROR, session=session, delay=0)
        session.urls.clear()
        text, from_cache = download_book(1, mirror=MIRROR, session=session, delay=0)
        assert (text, from_cache) == ("body", True)
        assert session.urls == []

    def test_missing_book_returns_none(self, data_root):
        session = FakeSession({})
        assert download_book(99, mirror=MIRROR, session=session, delay=0)[0] is None


class TestCollector:
    def _session(self):
        return FakeSession({
            f"{MIRROR}/0/1/1-0.txt": BOOK,
            f"{MIRROR}/0/2/2-0.txt": BOOK.replace('Ishmael', 'Ahab'),
        }, catalog=CATALOG)

    def test_writes_the_expected_columns(self, data_root):
        out = data_root / 'books.parquet'
        collect_gutenberg(str(out), mirror=MIRROR, session=self._session(),
                          delay=0, min_chars=10)
        df = pd.read_parquet(out)
        assert df.columns.tolist() == ['text', 'book_id', 'title', 'author']
        assert df['book_id'].tolist() == [1, 2]
        assert 'FULL LICENSE' not in ' '.join(df['text'])

    def test_max_books_caps_the_run(self, data_root):
        out = data_root / 'books.parquet'
        counts = collect_gutenberg(str(out), mirror=MIRROR, session=self._session(),
                                   delay=0, max_books=1, min_chars=10)
        assert counts['written'] == 1

    def test_unavailable_books_are_skipped_not_fatal(self, data_root):
        out = data_root / 'books.parquet'
        session = FakeSession({f"{MIRROR}/0/1/1-0.txt": BOOK}, catalog=CATALOG)
        counts = collect_gutenberg(str(out), mirror=MIRROR, session=session,
                                   delay=0, min_chars=10)
        assert counts == {'written': 1, 'cached': 0, 'skipped': 0, 'failed': 1}

    def test_short_books_are_skipped(self, data_root):
        out = data_root / 'books.parquet'
        counts = collect_gutenberg(str(out), mirror=MIRROR, session=self._session(),
                                   delay=0, min_chars=100000)
        assert counts['skipped'] == 2 and counts['written'] == 0

    def test_a_concurrent_run_is_refused(self, data_root):
        # Two collections sharing one output path interleave row groups and leave a
        # file whose footer parses but whose middle does not.
        out = data_root / 'books.parquet'
        lock = data_root / 'books.parquet.lock'
        lock.write_text('1')  # pid 1 always exists and is not this process
        with pytest.raises(RuntimeError, match="already writing"):
            collect_gutenberg(str(out), mirror=MIRROR, session=self._session(),
                              delay=0, min_chars=10)

    def test_a_stale_lock_is_taken_over(self, data_root):
        out = data_root / 'books.parquet'
        # A pid that cannot be running: the collection should proceed anyway.
        (data_root / 'books.parquet.lock').write_text('999999999')
        counts = collect_gutenberg(str(out), mirror=MIRROR, session=self._session(),
                                   delay=0, min_chars=10)
        assert counts['written'] == 2

    def test_the_lock_is_released_afterwards(self, data_root):
        out = data_root / 'books.parquet'
        collect_gutenberg(str(out), mirror=MIRROR, session=self._session(),
                          delay=0, min_chars=10)
        assert not (data_root / 'books.parquet.lock').exists()

    def test_a_failed_run_leaves_the_previous_corpus_intact(self, data_root):
        out = data_root / 'books.parquet'
        collect_gutenberg(str(out), mirror=MIRROR, session=self._session(),
                          delay=0, min_chars=10)
        before = out.read_bytes()

        class Boom(FakeSession):
            def get(self, url, **kw):
                if url.endswith('.csv.gz'):
                    return super().get(url, **kw)
                raise RuntimeError('mirror exploded')

        with pytest.raises(RuntimeError, match='mirror exploded'):
            collect_gutenberg(str(out), mirror=MIRROR,
                              session=Boom({}, catalog=CATALOG), delay=0,
                              min_chars=10, cache_dir=data_root / 'empty_cache')

        assert out.read_bytes() == before
        assert not (data_root / 'books.parquet.lock').exists()

    def test_no_partial_file_remains_after_success(self, data_root):
        out = data_root / 'books.parquet'
        collect_gutenberg(str(out), mirror=MIRROR, session=self._session(),
                          delay=0, min_chars=10)
        assert not list(data_root.glob('*.partial-*'))

    def test_output_is_fully_readable_end_to_end(self, data_root):
        # The corruption this guards against only shows up when every row group is
        # actually decoded, not when the footer is parsed.
        import pyarrow.parquet as pq
        out = data_root / 'books.parquet'
        collect_gutenberg(str(out), mirror=MIRROR, session=self._session(),
                          delay=0, min_chars=10)
        f = pq.ParquetFile(str(out))
        rows = sum(f.read_row_group(i).num_rows for i in range(f.metadata.num_row_groups))
        assert rows == f.metadata.num_rows == 2

    def test_rerun_uses_the_cache(self, data_root):
        out = data_root / 'books.parquet'
        collect_gutenberg(str(out), mirror=MIRROR, session=self._session(),
                          delay=0, min_chars=10)
        session = self._session()
        counts = collect_gutenberg(str(out), mirror=MIRROR, session=session,
                                   delay=0, min_chars=10)
        assert counts['cached'] == 2
        assert session.urls == []


class TestSampleParquetTexts:
    """
    Reading the corpus with pd.read_parquet held the Arrow buffers and the decoded
    Python strings at once, and the full Gutenberg corpus got t2-train OOM-killed at
    82 GB RSS. The sampler streams instead, so peak memory follows max_texts.
    """

    def _corpus(self, tmp_path, n):
        import pyarrow as pa
        import pyarrow.parquet as pq
        path = tmp_path / 'corpus.parquet'
        table = pa.table({'text': [f'chunk {i}' for i in range(n)]})
        pq.write_table(table, path, row_group_size=100)
        return str(path)

    def test_caps_at_max_texts(self, tmp_path):
        from transformer2.data.loaders import sample_parquet_texts
        assert len(sample_parquet_texts(self._corpus(tmp_path, 1000), 250)) == 250

    def test_zero_keeps_everything(self, tmp_path):
        from transformer2.data.loaders import sample_parquet_texts
        assert len(sample_parquet_texts(self._corpus(tmp_path, 1000), 0)) == 1000

    def test_max_above_corpus_size_keeps_everything(self, tmp_path):
        from transformer2.data.loaders import sample_parquet_texts
        assert len(sample_parquet_texts(self._corpus(tmp_path, 300), 5000)) == 300

    def test_sample_is_drawn_from_across_the_corpus(self, tmp_path):
        # Taking the first N rows would return only the earliest books.
        from transformer2.data.loaders import sample_parquet_texts
        got = sample_parquet_texts(self._corpus(tmp_path, 10000), 500)
        indices = [int(t.split()[1]) for t in got]
        assert max(indices) > 5000, "sample is confined to the front of the corpus"

    def test_rows_are_unique(self, tmp_path):
        from transformer2.data.loaders import sample_parquet_texts
        got = sample_parquet_texts(self._corpus(tmp_path, 2000), 400)
        assert len(set(got)) == len(got)

    def test_is_deterministic_for_a_seed(self, tmp_path):
        from transformer2.data.loaders import sample_parquet_texts
        path = self._corpus(tmp_path, 2000)
        assert sample_parquet_texts(path, 300) == sample_parquet_texts(path, 300)
