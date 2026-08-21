"""
Ingestion: reads raw .txt documents from data/corpus/, parses their metadata
header (TITLE / SOURCE / DATE), and splits the body into overlapping word-count
chunks.

Every chunk carries its source document's metadata forward. This matters
because contradiction detection happens at the chunk/claim level later in the
pipeline, but the final flagged output needs to say "Source A (2019) vs
Source B (2024)" — that requires source + date to still be attached by the
time we get there, not just known at ingestion time.
"""

import os
import re
from dataclasses import dataclass, field


@dataclass
class Chunk:
    chunk_id: str       # unique id: "<doc_id>__chunk<N>"
    doc_id: str          # filename without extension, e.g. "hr_handbook_2019"
    title: str
    source: str
    date: str
    chunk_index: int
    text: str


def _parse_document(raw_text: str, fallback_source: str = None, fallback_date: str = None) -> dict:
    """
    Splits a raw file's text into its header fields and body.
    Expected format:
        TITLE: ...
        SOURCE: ...
        DATE: ...
        <blank line>
        <body text>

    If the file doesn't match this format — which real-world uploaded
    documents from a company almost certainly won't, since this header is
    a convention specific to our seeded demo corpus — falls back to
    treating the whole file as body text, using fallback_source (typically
    the filename) and fallback_date (typically user-supplied at upload
    time, since we have no other way to know a real document's date).
    """
    header_pattern = re.compile(
        r"TITLE:\s*(?P<title>.+)\n"
        r"SOURCE:\s*(?P<source>.+)\n"
        r"DATE:\s*(?P<date>.+)\n\n"
        r"(?P<body>.*)",
        re.DOTALL,
    )
    match = header_pattern.match(raw_text.strip() + "\n")
    if match:
        return {
            "title": match.group("title").strip(),
            "source": match.group("source").strip(),
            "date": match.group("date").strip(),
            "body": match.group("body").strip(),
        }

    if fallback_source is None or fallback_date is None:
        raise ValueError(
            "Document does not match expected header format "
            "(TITLE / SOURCE / DATE / blank line / body), and no fallback "
            "source/date was provided."
        )
    return {
        "title": fallback_source,
        "source": fallback_source,
        "date": fallback_date,
        "body": raw_text.strip(),
    }


def _chunk_text(text: str, chunk_size: int = 200, overlap: int = 40) -> list[str]:
    """
    Splits text into word-count chunks with overlap.

    chunk_size and overlap are in words, not characters — word count is a
    reasonable proxy for "how much can an LLM call comfortably reason about
    as one claim-extraction unit" without needing an actual tokenizer here.

    For short documents (our seeded corpus), this returns a single chunk,
    since len(words) < chunk_size. The logic still needs to be correct for
    longer real-world documents, which is the point of writing it properly
    now rather than assuming short input.
    """
    words = text.split()
    if len(words) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        if end >= len(words):
            break
        start = end - overlap  # step back by `overlap` words for the next chunk
    return chunks


def load_corpus(
    corpus_dir: str | list[str],
    chunk_size: int = 200,
    overlap: int = 40,
    fallback_date: str = None,
) -> list[Chunk]:
    """
    Reads every .txt file across one or more corpus directories, parses
    each, chunks the body, and returns a flat list of Chunk objects ready
    for embedding.

    Accepting a list here (not just a single directory) is what lets the
    pipeline combine the seeded demo corpus with a user's uploaded
    documents, or run against just one or the other — a company checking
    their own policies almost certainly doesn't want fictional demo
    documents like "Medazol dosage" mixed into their real results.

    fallback_date is passed through to _parse_document for any file that
    doesn't match our header format — real uploaded files won't have it,
    and without a date, recency-gap severity scoring can't work for that
    document.
    """
    if isinstance(corpus_dir, str):
        corpus_dirs = [corpus_dir]
    else:
        corpus_dirs = corpus_dir

    all_chunks: list[Chunk] = []
    any_files_found = False

    for directory in corpus_dirs:
        if not os.path.isdir(directory):
            continue  # an optional directory (e.g. no uploads yet) simply contributes nothing

        filenames = sorted(f for f in os.listdir(directory) if f.endswith(".txt"))
        for filename in filenames:
            any_files_found = True
            doc_id = filename[:-4]  # strip ".txt"
            path = os.path.join(directory, filename)
            with open(path, "r", encoding="utf-8") as f:
                raw_text = f.read()

            parsed = _parse_document(raw_text, fallback_source=doc_id, fallback_date=fallback_date)
            body_chunks = _chunk_text(parsed["body"], chunk_size=chunk_size, overlap=overlap)

            for i, chunk_text in enumerate(body_chunks):
                all_chunks.append(
                    Chunk(
                        chunk_id=f"{doc_id}__chunk{i}",
                        doc_id=doc_id,
                        title=parsed["title"],
                        source=parsed["source"],
                        date=parsed["date"],
                        chunk_index=i,
                        text=chunk_text,
                    )
                )

    if not any_files_found:
        raise FileNotFoundError(f"No .txt files found in any of: {corpus_dirs}")

    return all_chunks


if __name__ == "__main__":
    # Quick manual check: run `python core/ingestion.py` from the project root
    # to confirm every doc in data/corpus/ parses and chunks without error.
    chunks = load_corpus("data/corpus")
    print(f"Loaded {len(chunks)} chunks from {len(set(c.doc_id for c in chunks))} documents.\n")
    for c in chunks[:3]:
        print(f"[{c.chunk_id}] source={c.source} date={c.date}")
        print(f"  text: {c.text[:80]}...\n")