from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable

from .models import Chunk, Decision

SECTION_RE = re.compile(r"(?:^|\n)(?P<title>[A-Z][^\n]{2,90})\n")
POLICY_HINTS = ("community standard", "content policy", "policy", "rule")


def clean_text(text: str) -> str:
    text = text.replace("\u00a0", " ").replace("\u200b", "")
    return re.sub(r"[ \t]+", " ", re.sub(r"\n{3,}", "\n\n", text)).strip()


def split_sections(text: str) -> dict[str, str]:
    matches = list(SECTION_RE.finditer(text))
    if not matches:
        return {"Decision": clean_text(text)}
    result: dict[str, str] = {}
    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = clean_text(text[start:end])
        if len(body) >= 80:
            result[clean_text(match.group("title"))] = body
    return result or {"Decision": clean_text(text)}


def sliding_chunks(text: str, size: int = 900, overlap: int = 120) -> Iterable[str]:
    words = text.split()
    if not words:
        return
    approx_words = max(50, size // 6)
    overlap_words = min(approx_words - 1, overlap // 6)
    step = max(1, approx_words - overlap_words)
    for start in range(0, len(words), step):
        part = " ".join(words[start : start + approx_words])
        if len(part) >= 100:
            yield part
        if start + approx_words >= len(words):
            break


def decision_to_chunks(decision: Decision, size: int = 900, overlap: int = 120) -> list[Chunk]:
    sections = decision.sections or split_sections(decision.full_text)
    chunks: list[Chunk] = []
    for section, body in sections.items():
        source_type = "policy" if any(h in section.lower() for h in POLICY_HINTS) else "precedent"
        for index, text in enumerate(sliding_chunks(body, size, overlap)):
            raw_id = f"{decision.case_id}:{section}:{index}"
            chunks.append(
                Chunk(
                    chunk_id=hashlib.sha1(raw_id.encode()).hexdigest()[:16],
                    case_id=decision.case_id,
                    source_type=source_type,
                    title=decision.title,
                    text=text,
                    url=str(decision.url),
                    section=section,
                    outcome=decision.outcome,
                    published_date=decision.published_date,
                    policies=decision.policies,
                )
            )
    return chunks
