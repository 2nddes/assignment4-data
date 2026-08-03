from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.encoding import detect_encoding
from typing import Tuple, Any, Dict, List
import fasttext
from fastwarc.warc import ArchiveIterator, WarcRecordType
import re
import random


def extract_text_from_html_bytes(html_bytes: bytes) -> str:
    """
    使用 resiliparse 从 HTML 字节串中提取纯文本
    :param html_bytes: 原始 HTML 字节串
    :param main_content_only: 是否开启正文提取机制（过滤导航栏、页脚等噪声）
    """
    if not html_bytes:
        return ""

    # 1. 优先尝试标准的 UTF-8 解码
    try:
        html_str = html_bytes.decode('utf-8')
    except UnicodeDecodeError:
        # 2. 解码失败时使用 Resiliparse 进行自动编码检测
        detected_enc = detect_encoding(html_bytes) or 'utf-8'
        try:
            html_str = html_bytes.decode(detected_enc, errors='replace')
        except (LookupError, UnicodeDecodeError):
            html_str = html_bytes.decode('utf-8', errors='replace')

    # 3. 提取纯文本 (可控制是否仅保留 main_content)
    return extract_plain_text(html_str)


def identify_language(text: str) -> tuple[str, float]:
    """
    Identifies the main language of a given Unicode string using fastText.

    Parameters:
        text (str): The input text to analyze.

    Returns:
        tuple[str, float]: A pair (language_code, confidence_score)
                           where language_code is an ISO 639 code (e.g., 'en', 'es', 'zh')
                           and confidence_score is a float between 0.0 and 1.0.
    """
    import fasttext

    MODEL_PATH = "local-shared-data/classifiers/lid.176.bin"

    model = fasttext.load_model(MODEL_PATH)

    if not model:
        raise RuntimeError("FastText model is not loaded. Please verify the file path.")

    cleaned_text = text.replace("\n", " ").strip()

    # fastText returns labels like ('__label__en',) and probabilities array([0.98])
    labels, probabilities = model.predict(cleaned_text, k=1)
    
    # Extract language code (strip the '__label__' prefix) and confidence score
    raw_label = labels[0]
    language_code = raw_label.replace("__label__", "")
    confidence_score = float(probabilities[0])

    return (language_code, round(confidence_score, 4))


def read_warc_records(warc_path: str) -> Dict[str, str]:
    """读取 WARC 文件中的 response 记录，并提取文本"""
    extracted_texts = []
    
    # 支持本地文件或文件流
    stream = open(warc_path, 'rb') if isinstance(warc_path, str) else warc_path
    
    for record in ArchiveIterator(stream, record_types=WarcRecordType.response):
        raw_payload = record.reader.read()
        extracted_text = extract_text_from_html_bytes(raw_payload)
        extracted_texts.append(extracted_text)
        
    return extracted_texts


def mask_emails(text: str) -> tuple[str, int]:
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'
    return re.subn(email_pattern, "|||EMAIL_ADDRESS|||", text)

def mask_phone_number(text: str) -> tuple[str, int]:
    phone_pattern = r'(?<!\d)(?:\+?1[-.\s]?)?(?:\(\d{3}\)|\d{3})[-.\s]?\d{3}[-.\s]?\d{4}(?!\d)'
    return re.subn(phone_pattern, "|||PHONE_NUMBER|||", text)

def mask_ip(text: str) -> tuple[str, int]:
    octet = r'(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)'
    ip_pattern = rf'\b(?:{octet}\.){{3}}{octet}\b'
    return re.subn(ip_pattern, "|||IP_ADDRESS|||", text)


def sampling_warc(counts: int) -> List:
    extracted_texts = read_warc_records("local-shared-data/CC/example.warc.gz")
    return random.sample(extracted_texts, counts)


NSFW_MODEL_PATH = "local-shared-data/classifiers/dolma_fasttext_nsfw_jigsaw_model.bin"
TOXIC_MODEL_PATH = "local-shared-data/classifiers/dolma_fasttext_hatespeech_jigsaw_model.bin"

from functools import lru_cache

@lru_cache(maxsize=1)
def get_nsfw_model():
    return fasttext.load_model(NSFW_MODEL_PATH)
@lru_cache(maxsize=1)
def get_toxic_model():
    return fasttext.load_model(TOXIC_MODEL_PATH)


def preprocess_text(text: str) -> str:
    return text.replace('\n', ' ').strip().lower()

def run_classify_nsfw(text: str) -> tuple[str, float]:
    nsfw_model = get_nsfw_model()

    cleaned_text = preprocess_text(text)
    
    labels, probabilities = nsfw_model.predict(cleaned_text, k=1)
    
    label = labels[0].replace("__label__", "")
    confidence = float(probabilities[0])

    return label, confidence

def run_classify_toxic_speech(text: str) -> tuple[str, float]:
    toxic_model = get_toxic_model()

    cleaned_text = preprocess_text(text)
    
    labels, probabilities = toxic_model.predict(cleaned_text, k=1)
    
    label = labels[0].replace("__label__", "")
    confidence = float(probabilities[0])

    return label, confidence

def gopher_filter(text: str) -> bool:
    if not text or not text.strip():
        return False

    import nltk

    nltk.data.path.insert(0, 'nltk_data')

    words = nltk.word_tokenize(text=text)

    if len(words) < 50 or len(words) > 100000:
        return False

    mean_length = sum(len(w) for w in words) / len(words)
    if mean_length > 10 or mean_length < 3:
        return False

    
    lines = text.strip().splitlines()
    if len(lines) > 0:
        ellipsis_lines = sum(1 for line in lines if line.strip().endswith("..."))
        if ellipsis_lines / len(lines) > 0.3:
            return False

    alpha_words = sum(1 for w in words if any(c.isalpha() for c in w))

    if (alpha_words / len(words)) < 0.80:
        return False
        
    return True