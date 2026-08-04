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


import os
import hashlib
from collections import defaultdict

def deduplicate_corpus(input_files: list[str], output_dir: str) -> None:
    """
    Deduplicates lines across a corpus of files by removing lines that occur 
    more than once in total.
    
    Args:
        input_files: A list of file paths to process.
        output_dir: The directory where the deduplicated files should be saved.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    freq_map = defaultdict(int)
    
    # Pass 1: Count line frequencies across the entire corpus
    for filepath in input_files:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                h = hashlib.md5(line.encode('utf-8')).digest()
                freq_map[h] += 1
                
    # Pass 2: Rewrite files, keeping only globally unique lines
    for filepath in input_files:
        # Extract the base filename (e.g., '1.txt' from 'a/1.txt')
        filename = os.path.basename(filepath)
        output_path = os.path.join(output_dir, filename)
        
        with open(filepath, 'r', encoding='utf-8') as f_in, \
             open(output_path, 'w', encoding='utf-8') as f_out:
            for line in f_in:
                h = hashlib.md5(line.encode('utf-8')).digest()
                # If the line occurred exactly once in the entire corpus, write it
                if freq_map[h] == 1:
                    f_out.write(line)

import shutil
import struct
import itertools
import unicodedata

def normalize_text(text: str) -> str:
    """
    Applies NFD unicode normalization, removes accents, removes punctuation,
    lowercases the text, and normalizes whitespace.
    """
    # NFD unicode normalization，把带音标的拆成字母加音标："Café" → "Cafe\u0301"     # é 拆成 e + ́
    text = unicodedata.normalize('NFD', text)
    # Remove accents (Nonspacing Marks)，去除重音
    text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
    # Lowercase
    text = text.lower()
    # Remove punctuation，去除标点
    text = re.sub(r'[^\w\s]', '', text)
    # Normalize whitespaces，多个连续空格 → 单个空格，首尾空格去除
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_word_ngrams(text: str, n: int) -> set:
    """
    Tokenizes text into word n-grams.

    例如:
    text = "cafe is a great place etudier a paris", 
    n = 2
    返回:
    {
        "cafe is", "is a", "a great", "great place",
        "place etudier", "etudier a", "a paris"
    }   # 共 7 个元素（set 类型，无序）
    """
    words = text.split()
    if not words:
        return set()
    if len(words) < n:
        return {' '.join(words)}
    
    return {' '.join(words[i:i+n]) for i in range(len(words) - n + 1)}

def compute_minhash_signature(ngrams: set, num_hashes: int) -> list:
    """
    Computes a MinHash signature for a given set of n-grams.
    """
    signature = [float('inf')] * num_hashes
    if not ngrams:
        return [0] * num_hashes
        
    for ngram in ngrams:
        encoded_ngram = ngram.encode('utf-8')
        for i in range(num_hashes):
            # Create a deterministic, unique hash function using salt 'i'
            hasher = hashlib.md5()
            hasher.update(str(i).encode('utf-8') + b'-' + encoded_ngram)
            
            # Unpack the first 4 bytes into an unsigned 32-bit integer
            hash_val = struct.unpack('<I', hasher.digest()[:4])[0]
            if hash_val < signature[i]:
                signature[i] = hash_val
                
    return signature

def deduplicate_documents(
    input_paths: list, 
    num_hashes: int, 
    num_bands: int, 
    ngram_size: int, 
    jaccard_threshold: float,
    output_dir: str
):
    """
    Deduplicates a list of files using MinHash and LSH, and writes the retained files 
    to the output directory.

    Args:
        input_paths: List of file paths to process.
        num_hashes: Length of the MinHash signature.
        num_bands: Number of bands to divide the signature into for LSH.
        ngram_size: The 'n' in n-gram length (in words).
        jaccard_threshold: The true Jaccard similarity threshold above which documents 
                           are considered duplicates.
        output_dir: Directory where the deduplicated files will be saved.
    """
    if num_hashes % num_bands != 0:
        raise ValueError("num_hashes must be evenly divisible by num_bands.")
    
    os.makedirs(output_dir, exist_ok=True)
    rows_per_band = num_hashes // num_bands

    doc_ngrams = {}
    signatures = {}

    # 1. Read files, normalize, extract n-grams, and compute MinHash signatures
    for path in input_paths:
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()
            
        norm_text = normalize_text(text)
        ngrams = get_word_ngrams(norm_text, ngram_size)
        
        doc_ngrams[path] = ngrams
        signatures[path] = compute_minhash_signature(ngrams, num_hashes)

    # 2. Locality Sensitive Hashing (LSH) to identify candidate pairs
    lsh_buckets = defaultdict(list)
    candidates = set()
    
    for path, sig in signatures.items():
        for band_idx in range(num_bands):
            start = band_idx * rows_per_band
            end = start + rows_per_band
            # A band is represented as a tuple of hash values
            band_tuple = tuple(sig[start:end])
            
            # Add a band prefix so bands don't collide across different parts of the signature
            bucket_key = (band_idx, band_tuple)
            lsh_buckets[bucket_key].append(path)

    # Generate candidate pairs from buckets that share at least one band
    for bucket_key, docs in lsh_buckets.items():
        if len(docs) > 1:
            for u, v in itertools.combinations(sorted(docs), 2):
                candidates.add((u, v))

    # 3. Filter candidates by true n-gram Jaccard similarity
    duplicate_edges = []
    for u, v in candidates:
        set_u, set_v = doc_ngrams[u], doc_ngrams[v]
        intersection = len(set_u & set_v)
        union = len(set_u | set_v)
        
        sim = intersection / union if union > 0 else 1.0
        if sim >= jaccard_threshold:
            duplicate_edges.append((u, v))

    # 4. Cluster duplicates into Connected Components
    # Treat paths as nodes and true duplicate relationships as undirected edges
    adj = {path: [] for path in input_paths}
    for u, v in duplicate_edges:
        adj[u].append(v)
        adj[v].append(u)

    visited = set()
    components = []
    
    for path in input_paths:
        if path not in visited:
            comp = []
            stack = [path]
            visited.add(path)
            
            while stack:
                curr = stack.pop()
                comp.append(curr)
                for neighbor in adj[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        stack.append(neighbor)
            components.append(comp)

    # 5. Randomly retain exactly one document from each component/cluster
    retained_paths = []
    for comp in components:
        # For non-duplicates, the component size is 1, so the document is inherently kept
        retained = random.choice(comp)
        retained_paths.append(retained)

    # 6. Write out the retained files to the output directory
    for path in retained_paths:
        filename = os.path.basename(path)
        out_path = os.path.join(output_dir, filename)
        shutil.copy2(path, out_path)


        