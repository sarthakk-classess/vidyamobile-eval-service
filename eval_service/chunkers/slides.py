"""
chunkers/slides.py — Lecture slides chunker.

Chunk unit  : group of 3–5 consecutive slides from the same topic
Target size : 400 – 900 characters (~100 – 225 tokens)
Overlap     : 1 slide at group boundaries
Min/Max     : 100 / 2,000 characters (hard limits)

Input: list of SlideInput objects from the PPTX parser (RT-05).
"""

from __future__ import annotations
import re
from dataclasses import dataclass

from eval_service.chunkers.base import BaseChunker, SlideChunk, ChunkError
from eval_service.chunkers.utils.id_utils import generate_chunk_id


@dataclass
class SlideInput:
    """Represents one parsed slide. RT-05 must produce this before calling chunk()."""
    slide_number: int
    title:        str
    body:         str


class SlideChunker(BaseChunker):
    """Groups lecture slides into topic-coherent chunks."""

    SOFT_GROUP_MAX = 4
    OVERLAP_SLIDES = 1
    STOP_WORDS = {
        'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
        'for', 'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were',
        'it', 'this', 'that', 'be', 'as', 'if', 'we', 'our', 'you', 'your'
    }

    def chunk(self, slides: list[SlideInput], doc_metadata: dict) -> list[dict]:
        if not slides:
            raise ChunkError("too_short", "No slides provided.")
        non_empty = [s for s in slides if s.title.strip() or s.body.strip()]
        if not non_empty:
            raise ChunkError("too_short", "All slides are empty (image-only deck).")

        groups    = self._group_slides(slides)
        overlapped = self._apply_overlap(groups)

        chunks = []
        for i, group in enumerate(overlapped):
            text       = self._format_group_text(group)
            titles     = [s.title for s in group if s.title.strip()]
            slide_nums = [s.slide_number for s in group]
            chunk_id   = generate_chunk_id(
                "lecture_slides",
                doc_metadata.get("document_path", doc_metadata.get("document_title", "unknown")),
                i,
            )
            chunk = SlideChunk(
                chunk_id       = chunk_id,
                document_title = doc_metadata.get("document_title", ""),
                lecture_name   = doc_metadata.get("lecture_name", doc_metadata.get("document_title", "")),
                week_number    = doc_metadata.get("week_number"),
                slide_range    = f"{min(slide_nums)}-{max(slide_nums)}",
                slide_start    = min(slide_nums),
                slide_end      = max(slide_nums),
                slide_titles   = titles,
                topic          = self._infer_topic(titles),
                char_count     = len(text),
                chunk_index    = i,
                text           = text,
            )
            chunks.append(chunk.to_dict())
        return chunks

    def _group_slides(self, slides: list[SlideInput]) -> list[list[SlideInput]]:
        groups  = []
        current = [slides[0]]
        for slide in slides[1:]:
            is_divider  = (not slide.title.strip() and not slide.body.strip())
            topic_shift = (
                len(current) >= 2 and
                self._is_topic_shift(slide.title, [s.title for s in current[-3:]])
            )
            at_max = len(current) >= self.SOFT_GROUP_MAX
            if is_divider or topic_shift or at_max:
                groups.append(current)
                current = [slide]
            else:
                current.append(slide)
        if current:
            groups.append(current)
        return groups

    def _is_topic_shift(self, new_title: str, recent_titles: list[str]) -> bool:
        if not new_title.strip():
            return False
        def keywords(title: str) -> set[str]:
            words = re.findall(r'[a-zA-Z]+', title.lower())
            return {w for w in words if w not in self.STOP_WORDS and len(w) >= 3}
        new_kws = keywords(new_title)
        if not new_kws:
            return False
        recent_kws: set[str] = set()
        for t in recent_titles:
            recent_kws |= keywords(t)
        return len(new_kws & recent_kws) == 0

    def _apply_overlap(self, groups: list[list[SlideInput]]) -> list[list[SlideInput]]:
        if len(groups) <= 1:
            return groups
        result = [groups[0]]
        for i in range(1, len(groups)):
            result.append(groups[i - 1][-self.OVERLAP_SLIDES:] + groups[i])
        return result

    def _format_group_text(self, group: list[SlideInput]) -> str:
        parts = []
        for slide in group:
            if not slide.title.strip() and not slide.body.strip():
                parts.append(f"Slide {slide.slide_number}: [diagram — no text]")
            elif not slide.body.strip():
                parts.append(f"Slide {slide.slide_number}: {slide.title.strip()}")
            else:
                parts.append(f"Slide {slide.slide_number}: {slide.title.strip()}\n{slide.body.strip()}")
        return '\n\n'.join(parts)

    def _infer_topic(self, titles: list[str]) -> str:
        cleaned = [t.strip() for t in titles if t.strip()]
        if not cleaned:
            return ""
        if len(cleaned) > 1:
            prefixes = [t.split(':')[0].strip() if ':' in t else t for t in cleaned]
            if len(set(prefixes)) == 1:
                return prefixes[0]
            word_lists = [re.findall(r'\w+', t.lower()) for t in cleaned]
            common = []
            for words in zip(*word_lists):
                if len(set(words)) == 1:
                    common.append(words[0])
                else:
                    break
            if common:
                return ' '.join(common).title()
        return cleaned[0]
