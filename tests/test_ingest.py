"""Tests for the ingest stage."""

import tempfile
from pathlib import Path

from kg_generator.ingest.loader import DataLoader, Document
from kg_generator.config import Language


def test_load_txt():
    loader = DataLoader()
    with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as f:
        f.write("Hello world. This is a test document.")
        tmp = Path(f.name)

    docs = loader._load_txt(tmp)
    assert len(docs) == 1
    assert "Hello world" in docs[0].content
    tmp.unlink()


def test_load_json():
    loader = DataLoader()
    content = '[{"text": "First doc", "id": "1"}, {"text": "Second doc", "id": "2"}]'
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        f.write(content)
        tmp = Path(f.name)

    docs = loader._load_json(tmp)
    assert len(docs) == 2
    assert docs[0].content == "First doc"
    assert docs[0].doc_id == "1"
    tmp.unlink()


def test_cleaner_normalizes_whitespace():
    from kg_generator.ingest.cleaner import TextCleaner
    doc = Document(content="  Hello   world!\n\nExtra  spaces.  ")
    cleaner = TextCleaner()
    result = cleaner.clean(doc)
    assert "  " not in result.content
    assert result.content == "Hello world!\n\nExtra spaces."


def test_vietnamese_cleaner_normalizes_nfc_and_preserves_diacritics():
    from kg_generator.ingest.cleaner import TextCleaner

    doc = Document(content="  Dữ\u0303 liệu tiếng Việt…\r\n\r\nGiữ nguyên dấu.  ")
    result = TextCleaner(Language.VIETNAMESE).clean(doc)

    assert result.content == "Dữ liệu tiếng Việt...\n\nGiữ nguyên dấu."
    assert "_" not in result.content


def test_vietnamese_sentence_chunker_preserves_complete_sentences():
    from kg_generator.ingest.chunker import SentenceChunker

    text = (
        "Hà Nội là thủ đô của Việt Nam. "
        "Thành phố có nhiều hồ đẹp. "
        "Huế từng là kinh đô của Việt Nam."
    )
    chunks = SentenceChunker(
        target_tokens=9,
        overlap_tokens=0,
        language=Language.VIETNAMESE,
    ).chunk([Document(content=text, source="vi.txt", doc_id="vi")])

    assert len(chunks) >= 2
    assert all(chunk.content.endswith((".", "!", "?")) for chunk in chunks)
    assert all("_" not in chunk.content for chunk in chunks)
    assert [chunk.metadata["chunk_index"] for chunk in chunks] == list(range(len(chunks)))


def test_semantic_chunker_splits_at_topic_shift_with_fake_encoder():
    from kg_generator.ingest.chunker import SemanticChunker

    text = (
        "Mèo thích ngủ trong nhà. "
        "Mèo thường chơi vào buổi tối. "
        "Tên lửa đưa vệ tinh lên quỹ đạo."
    )
    encoder = lambda _texts: [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]]
    chunks = SemanticChunker(
        target_tokens=20,
        overlap_tokens=0,
        language=Language.ENGLISH,
        similarity_threshold=0.5,
        encoder=encoder,
    ).chunk([Document(content=text, source="topics.txt", doc_id="topics")])

    assert len(chunks) == 2
    assert "Mèo" in chunks[0].content
    assert chunks[1].content == "Tên lửa đưa vệ tinh lên quỹ đạo."
