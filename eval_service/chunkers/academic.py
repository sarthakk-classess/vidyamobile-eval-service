"""
chunkers/academic.py — Academic readings chunker (papers, textbook chapters).

Chunk unit  : one sub-section (level 2 or 3 heading) with prose and code
Target size : 800 – 1,600 characters (~200 – 400 tokens)
Overlap     : 200 characters (always ending at a sentence boundary)
Min/Max     : 100 / 2,000 characters (hard limits)

Code blocks are NEVER split — if a code block straddles a boundary,
the split moves to before or after it.
"""

from __future__ import annotations
import re

from eval_service.chunkers.base import BaseChunker, AcademicChunk, ChunkError
from eval_service.chunkers.utils.text_utils import (
    clean_text, has_code_block, protect_code_blocks,
)
from eval_service.chunkers.utils.id_utils import generate_chunk_id


class AcademicChunker(BaseChunker):
    """Chunks academic readings into sub-section-sized pieces."""

    TARGET_MIN = 800
    TARGET_MAX = 1600
    OVERLAP    = 200

    def chunk(self, raw_text: str, doc_metadata: dict) -> list[dict]:
        text = clean_text(raw_text)
        if len(text) < self.MIN_CHARS:
            raise ChunkError("too_short", f"Academic text is only {len(text)} chars after cleaning.")

        sections       = self._split_on_headers(text)
        sections       = self._merge_short_sections(sections)
        final_sections = self._split_long_sections(sections)

        chunks      = []
        para_cursor = 0
        for i, (section_text, section_title, section_number) in enumerate(final_sections):
            chunk_id = generate_chunk_id(
                "academic_reading",
                doc_metadata.get("document_path", doc_metadata.get("document_title", "unknown")),
                i,
            )
            chunk = AcademicChunk(
                chunk_id        = chunk_id,
                document_title  = doc_metadata.get("document_title", ""),
                author          = doc_metadata.get("author"),
                section_title   = section_title,
                section_number  = section_number,
                paragraph_index = para_cursor,
                page_number     = self._extract_page_number(section_text),
                has_code        = has_code_block(section_text),
                char_count      = len(section_text),
                chunk_index     = i,
                text            = section_text,
            )
            chunks.append(chunk.to_dict())
            para_cursor += section_text.count('\n\n') + 1
        return chunks

    def _split_on_headers(self, text: str) -> list[tuple[str, str, str | None]]:
        lines: list[str]                                = text.split('\n')
        sections: list[tuple[str, str, str | None]]     = []
        current_lines: list[str]                        = []
        current_title  = ""
        current_number: str | None = None

        for line in lines:
            header = self._detect_header(line)
            if header and current_lines:
                s = '\n'.join(current_lines).strip()
                if s:
                    sections.append((s, current_title, current_number))
                current_lines  = [line]
                current_title  = header["title"]
                current_number = header["number"]
            else:
                current_lines.append(line)

        if current_lines:
            s = '\n'.join(current_lines).strip()
            if s:
                sections.append((s, current_title, current_number))

        if len(sections) <= 1:
            blocks   = [b.strip() for b in text.split('\n\n') if b.strip()]
            sections = [(b, "", None) for b in blocks]

        return sections

    def _detect_header(self, line: str) -> dict | None:
        stripped = line.strip()
        if not stripped:
            return None
        md = re.match(r'^(#{1,3})\s+(.+)', stripped)
        if md:
            title  = md.group(2).strip()
            number = re.match(r'^(\d+(?:\.\d+)*)', title)
            return {"title": title, "number": number.group(1) if number else None}
        num = re.match(r'^(\d+(?:\.\d+){1,3})\s+([A-Z].{2,})', stripped)
        if num:
            return {"title": stripped, "number": num.group(1)}
        if re.match(r'^[A-Z][A-Z\s]{4,}$', stripped) and len(stripped) < 60:
            return {"title": stripped, "number": None}
        return None

    def _merge_short_sections(
        self, sections: list[tuple[str, str, str | None]]
    ) -> list[tuple[str, str, str | None]]:
        if not sections:
            return []
        result = []
        buf_text, buf_title, buf_num = sections[0]
        for (text, title, number) in sections[1:]:
            if len(buf_text) < self.TARGET_MIN:
                buf_text = (buf_text + "\n\n" + text).strip()
            else:
                result.append((buf_text, buf_title, buf_num))
                buf_text, buf_title, buf_num = text, title, number
        if buf_text:
            if result and len(buf_text) < self.MIN_CHARS:
                prev_text, prev_title, prev_num = result[-1]
                result[-1] = ((prev_text + "\n\n" + buf_text).strip(), prev_title, prev_num)
            else:
                result.append((buf_text, buf_title, buf_num))
        return result

    def _split_long_sections(
        self, sections: list[tuple[str, str, str | None]]
    ) -> list[tuple[str, str, str | None]]:
        result = []
        for (text, title, number) in sections:
            if len(text) <= self.TARGET_MAX:
                result.append((text, title, number))
                continue
            sub_texts = self._split_respecting_code(text)
            for j, sub in enumerate(sub_texts):
                sub_title = f"{title} (part {j+1})" if len(sub_texts) > 1 and title else title
                result.append((sub, sub_title, number))
        return result

    def _split_respecting_code(self, text: str) -> list[str]:
        protected  = protect_code_blocks(text)
        paragraphs = self._split_paragraphs_safe(text, protected)

        chunks  = []
        current = []
        cur_len = 0

        for para in paragraphs:
            if cur_len + len(para) > self.TARGET_MAX and current:
                chunks.append('\n\n'.join(current).strip())
                overlap_paras: list[str] = []
                overlap_len = 0
                for p in reversed(current):
                    if overlap_len + len(p) <= self.OVERLAP:
                        overlap_paras.insert(0, p)
                        overlap_len += len(p)
                    else:
                        break
                current = overlap_paras + [para]
                cur_len = sum(len(p) for p in current)
            else:
                current.append(para)
                cur_len += len(para)

        if current:
            chunk_text = '\n\n'.join(current).strip()
            if len(chunk_text) >= self.MIN_CHARS:
                chunks.append(chunk_text)
            elif chunks:
                chunks[-1] += '\n\n' + chunk_text
            else:
                chunks.append(chunk_text)

        return [c for c in chunks if len(c) >= self.MIN_CHARS] or [text]

    def _split_paragraphs_safe(self, text: str, protected: str) -> list[str]:
        paras: list[str] = []
        current_start = 0
        i = 0
        while i < len(protected) - 1:
            if protected[i] == '\n' and protected[i + 1] == '\n':
                if protected[i] != ' ':
                    para = text[current_start:i].strip()
                    if para:
                        paras.append(para)
                    while i < len(text) and text[i] == '\n':
                        i += 1
                    current_start = i
                    continue
            i += 1
        last = text[current_start:].strip()
        if last:
            paras.append(last)
        return paras or [text]

    def _extract_page_number(self, text: str) -> int | None:
        match = re.search(r'(?:^|\n)\s*[Pp]age\s+(\d+)', text)
        if match:
            return int(match.group(1))
        match = re.match(r'^\s*(\d{1,3})\s*\n', text)
        if match and int(match.group(1)) < 500:
            return int(match.group(1))
        return None
