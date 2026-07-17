"""Tests for the ingest stage."""

import tempfile
from pathlib import Path

from kg_generator.ingest.loader import DataLoader, Document


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
