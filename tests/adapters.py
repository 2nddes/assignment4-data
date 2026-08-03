from __future__ import annotations

import os
from typing import Any



def run_extract_text_from_html_bytes(html_bytes: bytes) -> str | None:
    from answers.utils import extract_text_from_html_bytes
    return extract_text_from_html_bytes(html_bytes=html_bytes)


def run_identify_language(text: str) -> tuple[Any, float]:
    from answers.utils import identify_language
    return identify_language(text=text)


def run_mask_emails(text: str) -> tuple[str, int]:
    from answers.utils import mask_emails
    return mask_emails(text=text)


def run_mask_phone_numbers(text: str) -> tuple[str, int]:
    from answers.utils import mask_phone_number
    return mask_phone_number(text=text)


def run_mask_ips(text: str) -> tuple[str, int]:
    from answers.utils import mask_ip
    return mask_ip(text=text)


def run_classify_nsfw(text: str) -> tuple[Any, float]:
    from answers.utils import run_classify_nsfw
    return run_classify_nsfw(text=text)


def run_classify_toxic_speech(text: str) -> tuple[Any, float]:
    from answers.utils import run_classify_toxic_speech
    return run_classify_toxic_speech(text=text)

def run_classify_quality(text: str) -> tuple[Any, float]:
    raise NotImplementedError


def run_gopher_quality_filter(text: str) -> bool:
    from answers.utils import gopher_filter
    return gopher_filter(text=text)


def run_exact_line_deduplication(
    input_files: list[os.PathLike], output_directory: os.PathLike
):
    raise NotImplementedError


def run_minhash_deduplication(
    input_files: list[os.PathLike],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: os.PathLike,
):
    raise NotImplementedError
