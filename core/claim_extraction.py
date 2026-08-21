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
import re
import time
from dataclasses import dataclass

import groq
from dotenv import load_dotenv
from groq import Groq

from core.ingestion import Chunk

load_dotenv()  # reads .env into environment variables — this is what keeps
                # the API key out of source code and out of git entirely

# openai/gpt-oss-120b replaces llama-3.3-70b-versatile, which Groq
# deprecated (email sent June 17, 2026; shut off for free/dev tier accounts).
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


def _extract_retry_seconds(error_message: str) -> float | None:
    """
    Groq's rate-limit error messages include their own suggested wait time,
    e.g. "Please try again in 659.999999ms." — parsing that out means we
    wait exactly as long as Groq says is needed, rather than guessing with
    a fixed or exponential delay that might be too short (fails again
    immediately) or too long (wastes time we don't have during a hackathon).
    """
    match = re.search(r"try again in ([\d.]+)ms", error_message)
    if match:
        return float(match.group(1)) / 1000.0
    match = re.search(r"try again in ([\d.]+)s", error_message)
    if match:
        return float(match.group(1))
    return None


def call_groq_with_retry(messages: list[dict], temperature: float = 0.1, max_retries: int = 5):
    """
    Shared wrapper for every Groq call in the pipeline (used by both
    claim_extraction.py and entailment.py). Free/dev-tier accounts have a
    tokens-per-minute cap that a ~50-call pipeline run can hit, especially
    after several full runs in the same session (exactly what happened
    during testing). Rather than letting one transient 429 kill the entire
    background thread and surface as a dead-end error to the user, this
    catches RateLimitError, waits the time Groq itself suggests (falling
    back to exponential backoff if that's not parseable), and retries.
    """
    client = get_client()
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                temperature=temperature,
            )
        except groq.RateLimitError as e:
            wait = _extract_retry_seconds(str(e))
            if wait is None:
                wait = 2 ** attempt  # exponential fallback if we can't parse Groq's suggestion
            wait = min(wait + 0.2, 30)  # small safety buffer, capped so we never wait absurdly long
            print(f"  Rate limited (attempt {attempt + 1}/{max_retries}), waiting {wait:.1f}s...")
            time.sleep(wait)
    raise RuntimeError(
        f"Exceeded {max_retries} retries due to Groq rate limiting. "
        f"Wait a minute for the tokens-per-minute window to reset, then try again."
    )


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
    prompt = EXTRACTION_PROMPT.format(text=chunk.text)

    response = call_groq_with_retry(
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