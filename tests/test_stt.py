"""
tests/test_stt.py
──────────────────
Basic STT smoke test — ensures WhisperTranscriber loads and can
strip the wake word from a transcript.
"""

import pytest
from mayuri.stt.transcriber import WhisperTranscriber


def test_strip_wake_word() -> None:
    t = WhisperTranscriber.__new__(WhisperTranscriber)
    assert t._strip_wake_word("Mayuri, open Chrome") == "open Chrome"
    assert t._strip_wake_word("Hey Mayuri open Chrome") == "open Chrome"
    assert t._strip_wake_word("mayuri. set the volume to 50") == "set the volume to 50"
    assert t._strip_wake_word("Hey Jarvis, what is the weather?") == "what is the weather?"
    assert t._strip_wake_word("Jarvis, search Amazon for laptops") == "search Amazon for laptops"
    assert t._strip_wake_word("search Amazon for laptops") == "search Amazon for laptops"
