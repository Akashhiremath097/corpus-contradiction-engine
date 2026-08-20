"""
Claim extraction: reduces each chunk to atomic, checkable assertions —
one claim per unit, per your deck's core insight (slide 5). Contradictions
can only exist between claims about the same thing, so breaking a chunk
like "employees must fast 8 hours, no clear liquids in the final window"
into separate atomic claims means each one can be compared and scored
independently later, instead of the whole chunk being one indivisible blob.

This is the first of two Groq-backed stages (the second is entailment.py).
"""

import json
import os
from dataclasses import dataclass

from dotenv import load_dotenv
from groq import Groq

from core.ingestion import Chunk

load_dotenv()  # reads .env into environment variables — this is what keeps
                # the API key out of source code and out of git entirely

GROQ_MODEL = "openai/gpt-oss-120b"

_client = None


def get_client() -> Groq:
    """
    Lazy singleton, same reasoning as the embedding model: creating a Groq
    client isn't free, and every function that needs one should share a
    single instance rather than constructing a new one per call.
    """
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY not set. Add it to a .env file in the project "
                "root: GROQ_API_KEY=your_key_here"
            )
        _client = Groq(api_key=api_key)
    return _client


@dataclass
class Claim:
    claim_id: str    # "<chunk_id>__claim<N>"
    chunk_id: str
    doc_id: str
    source: str
    date: str
    text: str


EXTRACTION_PROMPT = """You are extracting atomic factual claims from a policy or guideline document.

Break the following text into a list of atomic, checkable claims. Each claim should:
- State exactly one fact or rule
- Be self-contained and understandable without the rest of the text
- Preserve specific numbers, timeframes, and conditions exactly as written
- Contain only factual assertions, not opinions or reasoning

Return ONLY a JSON array of strings, nothing else. No markdown fences, no preamble, no explanation.

Text:
\"\"\"{text}\"\"\"

JSON array of claims:"""


def _strip_markdown_fences(text: str) -> str:
    """
    LLMs wrap JSON in ```json ... ``` fences surprisingly often, even when
    explicitly told not to. This strips any fence lines before parsing,
    so a stray ``` doesn't crash json.loads on an otherwise-valid response.
    """
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [line for line in lines if not line.strip().startswith("```")]
        text = "\n".join(lines).strip()
    return text


def extract_claims_from_chunk(chunk: Chunk) -> list[Claim]:
    """
    Calls Groq once per chunk to break it into atomic claims.

    temperature=0.1: extraction should be literal and repeatable — a low
    temperature keeps the model close to "restate what's actually there"
    rather than paraphrasing creatively, which matters because we need the
    specific numbers (500mg, 3 days, 24 hours) preserved exactly for the
    entailment step to catch contradictions correctly.
    """
    client = get_client()
    prompt = EXTRACTION_PROMPT.format(text=chunk.text)

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )

    raw = response.choices[0].message.content.strip()
    cleaned = _strip_markdown_fences(raw)

    try:
        claim_texts = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Model did not return valid JSON for chunk {chunk.chunk_id}. "
            f"Raw response (first 200 chars): {raw[:200]}"
        ) from e

    claims = []
    for i, claim_text in enumerate(claim_texts):
        claims.append(
            Claim(
                claim_id=f"{chunk.chunk_id}__claim{i}",
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                source=chunk.source,
                date=chunk.date,
                text=claim_text,
            )
        )
    return claims


def extract_claims_from_corpus(chunks: list[Chunk]) -> list[Claim]:
    """
    Runs extraction across every chunk, printing progress as it goes —
    with 20 chunks this takes a noticeable number of seconds (one API call
    each), so visible progress matters more than it would for a single call.
    """
    all_claims: list[Claim] = []
    for chunk in chunks:
        claims = extract_claims_from_chunk(chunk)
        all_claims.extend(claims)
        print(f"  {chunk.chunk_id}: {len(claims)} claim(s)")
    return all_claims


if __name__ == "__main__":
    from core.ingestion import load_corpus

    chunks = load_corpus("data/corpus")
    print(f"Extracting claims from {len(chunks)} chunks (one Groq call each)...\n")
    claims = extract_claims_from_corpus(chunks)

    print(f"\nTotal claims extracted: {len(claims)}")
    print("\nSample claims:")
    for c in claims[:5]:
        print(f"  [{c.claim_id}] {c.text}")