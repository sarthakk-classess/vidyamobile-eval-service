"""
Unit tests for chunkers (deterministic IDs, min/max lengths, field presence).
"""
import pytest
from eval_service.chunkers.base import SyllabusChunk, SlideChunk, AcademicChunk
from eval_service.chunkers.utils.id_utils import generate_chunk_id


def test_chunk_id_deterministic():
    id1 = generate_chunk_id("syllabus", "docs/syllabus.pdf", 0)
    id2 = generate_chunk_id("syllabus", "docs/syllabus.pdf", 0)
    assert id1 == id2
    assert id1.startswith("syl_")


def test_chunk_id_varies_by_path():
    id_a = generate_chunk_id("syllabus", "tenant_a/syllabus.pdf", 0)
    id_b = generate_chunk_id("syllabus", "tenant_b/syllabus.pdf", 0)
    assert id_a != id_b


def test_chunk_id_varies_by_index():
    id0 = generate_chunk_id("syllabus", "doc.pdf", 0)
    id1 = generate_chunk_id("syllabus", "doc.pdf", 1)
    assert id0 != id1


def test_slide_chunk_id_prefix():
    cid = generate_chunk_id("lecture_slides", "slides.pptx", 2)
    assert cid.startswith("sld_")


def test_academic_chunk_id_prefix():
    cid = generate_chunk_id("academic_reading", "paper.pdf", 0)
    assert cid.startswith("acad_")


def test_syllabus_chunk_fields():
    c = SyllabusChunk(chunk_id="syl_abc_0000", document_title="Syllabus 2026", text="Some content here.")
    assert c.doc_type == "syllabus"
    assert c.document_title == "Syllabus 2026"
    d = c.to_dict()
    assert "chunk_id" in d and "text" in d


def test_slide_chunk_fields():
    c = SlideChunk(chunk_id="sld_abc_0003", document_title="COS217 Lecture 8", slide_start=3, slide_end=6, text="content")
    assert c.doc_type == "lecture_slides"
    assert c.slide_start == 3


def test_academic_chunk_fields():
    c = AcademicChunk(chunk_id="acad_abc_0001", document_title="Paper", section_title="Abstract", text="abstract body")
    assert c.doc_type == "academic_reading"
    assert c.section_title == "Abstract"


def test_syllabus_chunker_basic(sample_syllabus_text):
    from eval_service.chunkers.syllabus import SyllabusChunker
    chunker = SyllabusChunker()
    chunks  = chunker.chunk(sample_syllabus_text, {"document_title": "Test Syllabus", "document_path": "test/syl.pdf"})
    assert isinstance(chunks, list)
    assert len(chunks) >= 1
    for c in chunks:
        assert "chunk_id" in c
        assert "text" in c


def test_academic_chunker_basic(sample_academic_text):
    from eval_service.chunkers.academic import AcademicChunker
    chunker = AcademicChunker()
    chunks  = chunker.chunk(sample_academic_text, {"document_title": "Test Paper", "document_path": "test/paper.pdf"})
    assert isinstance(chunks, list)
    assert len(chunks) >= 1
