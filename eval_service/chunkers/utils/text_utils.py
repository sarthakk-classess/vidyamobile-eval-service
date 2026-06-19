"""
Shared text processing utilities used by syllabus, slide, and academic chunkers.
Split boundaries and overlap values were calibrated against real university documents.
"""

from __future__ import annotations
import re
from typing import Optional


def find_sentence_boundary(text: str, near_pos: int, direction: str = "before") -> int:
    """Find the nearest sentence boundary to `near_pos`."""
    search_window = 200
    sentence_end  = re.compile(r'[.!?][\s]+(?=[A-Z0-9\-])')

    if direction == "before":
        start   = max(0, near_pos - search_window)
        segment = text[start:near_pos]
        matches = list(sentence_end.finditer(segment))
        if matches:
            return start + matches[-1].end() - 1
        return near_pos
    else:
        end     = min(len(text), near_pos + search_window)
        segment = text[near_pos:end]
        match   = sentence_end.search(segment)
        if match:
            return near_pos + match.end() - 1
        return near_pos


def split_at_sentence_boundary(text: str, max_chars: int, overlap: int) -> list[str]:
    """Split text into max_chars-sized pieces at sentence boundaries with overlap."""
    if len(text) <= max_chars:
        return [text]

    protected = protect_code_blocks(text)
    chunks    = []
    pos       = 0

    while pos < len(protected):
        end = min(pos + max_chars, len(protected))
        if end >= len(protected):
            chunk = text[pos:]
            if len(chunk) >= 100:
                chunks.append(chunk.strip())
            break
        boundary = find_sentence_boundary(protected, end, direction="before")
        if boundary <= pos:
            boundary = end
        chunks.append(text[pos:boundary].strip())
        next_start = max(pos + 1, boundary - overlap)
        next_start = find_sentence_boundary(protected, next_start, direction="after")
        pos = next_start

    return [c for c in chunks if len(c) >= 100]


def merge_short_texts(texts: list[str], min_chars: int = 100) -> list[str]:
    """Merge consecutive text segments below min_chars with the following segment."""
    if not texts:
        return []

    result = []
    buffer = ""

    for text in texts:
        if len(buffer) + len(text) < min_chars:
            buffer = (buffer + "\n\n" + text).strip() if buffer else text.strip()
        elif len(buffer) > 0 and len(buffer) < min_chars:
            result.append((buffer + "\n\n" + text).strip())
            buffer = ""
        else:
            if buffer:
                result.append(buffer.strip())
            buffer = text.strip()

    if buffer:
        if result and len(buffer) < min_chars:
            result[-1] = (result[-1] + "\n\n" + buffer).strip()
        else:
            result.append(buffer.strip())

    return [r for r in result if r]


def protect_code_blocks(text: str) -> str:
    """Replace code block contents with non-breaking space placeholders (same length)."""
    result = list(text)

    fence_pattern = re.compile(r'```[\s\S]*?```', re.DOTALL)
    for match in fence_pattern.finditer(text):
        for i in range(match.start(), match.end()):
            result[i] = ' '

    lines    = text.split('\n')
    char_pos = 0
    for line in lines:
        line_end = char_pos + len(line) + 1
        if re.match(r'^    ', line):
            for i in range(char_pos, min(line_end, len(result))):
                result[i] = ' '
        char_pos = line_end

    return ''.join(result)


def has_code_block(text: str) -> bool:
    """Returns True if the text contains a code block."""
    if re.search(r'```[\s\S]*?```', text, re.DOTALL):
        return True
    return any(re.match(r'^    \S', line) for line in text.split('\n'))


def detect_section_header(line: str) -> Optional[str]:
    """Detect whether a line is a section header; returns header text or None."""
    patterns = [
        r'^#{1,3}\s+(.+)',
        r'^(\d+(?:\.\d+)*)\s+[A-Z].{3,}',
        r'^(Unit\s+(?:I{1,3}V?|IV|V|VI|VII|VIII))',
        r'^(SEMESTER-[IVX]+)',
        r'^(\d[A-Z]\d[A-Z]+\d{2}[A-Z])',
    ]
    for pattern in patterns:
        match = re.match(pattern, line.strip())
        if match:
            return match.group(1) if match.lastindex else line.strip()
    return None


def extract_unit_numbers(text: str) -> list[str]:
    """Extract all unit references mentioned in text (deduplicated, order-preserved)."""
    pattern = re.compile(r'\bUnit\s+(I{1,3}V?|IV|V|VI|VII|VIII)\b')
    return list(dict.fromkeys(pattern.findall(text)))


def clean_text(text: str) -> str:
    """Basic cleaning applied before chunking: collapse 3+ newlines, strip whitespace."""
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()
