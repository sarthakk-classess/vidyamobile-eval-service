"""
chunkers/syllabus.py — Syllabus document chunker.

Chunk unit  : one logical section (course block, assessment section, week entry)
Target size : 600 – 1,200 characters (~150 – 300 tokens)
Overlap     : 150 characters
Min/Max     : 100 / 2,000 characters (hard limits)
"""

from __future__ import annotations
import re

from eval_service.chunkers.base import BaseChunker, SyllabusChunk, ChunkError
from eval_service.chunkers.utils.text_utils import (
    clean_text, extract_unit_numbers, merge_short_texts,
)
from eval_service.chunkers.utils.id_utils import generate_chunk_id


class SyllabusChunker(BaseChunker):
    """Chunks a syllabus document into searchable pieces."""

    TARGET_MIN = 600
    TARGET_MAX = 1200
    OVERLAP    = 150

    BOUNDARY_PATTERNS = [
        re.compile(r'^\d[A-Z]\d[A-Z]+\d{2}[A-Z]'),
        re.compile(r'^Unit\s+(?:I{1,3}V?|IV|V|VI|VII|VIII)\b', re.IGNORECASE),
        re.compile(r'^SEMESTER-[IVX]+', re.IGNORECASE),
        re.compile(r'^(?:Formative|Summative|ASSESSMENT\s+METHODS?)', re.IGNORECASE),
        re.compile(r'^(?:Year\s+\w+\s+Course\s+Code)', re.IGNORECASE),
        re.compile(r'^(?:Recommended|Reference)\s+(?:Learning\s+)?Resources?', re.IGNORECASE),
    ]

    def chunk(self, raw_text: str, doc_metadata: dict) -> list[dict]:
        text = clean_text(raw_text)
        if len(text) < self.MIN_CHARS:
            raise ChunkError("too_short", f"Syllabus text is only {len(text)} chars after cleaning.")

        sections = self._split_on_boundaries(text)
        sections = merge_short_texts(sections, min_chars=self.TARGET_MIN)
        sections = self.enforce_limits(sections, self.OVERLAP, self.TARGET_MAX)

        chunks = []
        for i, section_text in enumerate(sections):
            chunk_id = generate_chunk_id(
                "syllabus",
                doc_metadata.get("document_path", doc_metadata.get("document_title", "unknown")),
                i,
            )
            chunk = SyllabusChunk(
                chunk_id      = chunk_id,
                document_title = doc_metadata.get("document_title", ""),
                university     = doc_metadata.get("university"),
                semester       = self._extract_semester(section_text),
                course_code    = self._extract_course_code(section_text),
                course_title   = self._extract_course_title(section_text),
                section_type   = self._classify_section(section_text),
                unit_numbers   = extract_unit_numbers(section_text),
                char_count     = len(section_text),
                chunk_index    = i,
                text           = section_text,
            )
            chunks.append(chunk.to_dict())
        return chunks

    def _split_on_boundaries(self, text: str) -> list[str]:
        lines    = text.split('\n')
        sections = []
        current  = []
        for line in lines:
            if any(p.match(line.strip()) for p in self.BOUNDARY_PATTERNS) and current:
                s = '\n'.join(current).strip()
                if s:
                    sections.append(s)
                current = [line]
            else:
                current.append(line)
        if current:
            s = '\n'.join(current).strip()
            if s:
                sections.append(s)
        if len(sections) <= 1:
            sections = [s.strip() for s in text.split('\n\n') if s.strip()]
        return sections

    def _extract_semester(self, text: str) -> int | None:
        match = re.search(r'SEMESTER[-\s]*([I1-9]{1,3})', text, re.IGNORECASE)
        if not match:
            match = re.search(r'Sem(?:ester)?[.\s]*([12])', text, re.IGNORECASE)
        if match:
            val = match.group(1).upper()
            if val in ('I', '1'):  return 1
            if val in ('II', '2'): return 2
        return None

    def _extract_course_code(self, text: str) -> str | None:
        match = re.search(r'\b(\d[A-Z]\d[A-Z]+\d{2}[A-Z])\b', text)
        return match.group(1) if match else None

    def _extract_course_title(self, text: str) -> str | None:
        match = re.search(r'Course\s+Title:\s*(.+)', text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        for line in text.strip().split('\n')[:5]:
            line = line.strip()
            if 20 < len(line) < 100 and line[0].isupper() and ':' not in line:
                return line
        return None

    def _classify_section(self, text: str) -> str:
        lower = text.lower()
        if re.search(r'\b(cia|formative|summative|internal\s+assessment|sea|marks?)\b', lower):
            return "assessment"
        if re.search(r'\b(course\s+code|course\s+title|unit\s+[iv]|credits?\s*:\s*\d)\b', lower):
            return "course_block"
        if re.search(r'\b(recommended|reference\s+books?|bibliography)\b', lower):
            return "references"
        if re.search(r'\b(semester[-\s]*[i]{1,3}v?|schedule|timetable)\b', lower):
            return "schedule"
        if re.search(r'\b(preamble|introduction|objectives?|program\s+outcome)\b', lower):
            return "preamble"
        return "prose_block"
