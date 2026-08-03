import os
import gzip
import io
import requests
from typing import Dict, Tuple

from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.encoding import detect_encoding
from fastwarc.warc import ArchiveIterator, WarcRecordType


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


def read_warc_records(warc_path: str) -> Dict[str, str]:
    """读取 WARC 文件中的 response 记录，并提取文本"""
    extracted_texts = {}
    
    # 支持本地文件或文件流
    stream = open(warc_path, 'rb') if isinstance(warc_path, str) else warc_path
    
    for record in ArchiveIterator(stream, record_types=WarcRecordType.response):
        target_uri = record.headers.get('WARC-Target-URI')
        if not target_uri:
            continue
        
        # 仅处理 HTML 响应
        content_type = record.http_headers.get('Content-Type', '') if record.http_headers else ''
        if not content_type or 'text/html' not in content_type:
            continue

        raw_payload = record.reader.read()
        extracted_text = extract_text_from_html_bytes(raw_payload)
        extracted_texts[target_uri] = extracted_text
        
    return extracted_texts


def read_wet_records(wet_path: str) -> Dict[str, str]:
    """读取 WET 文件中的 conversion 记录（即 CC 预提取文本）"""
    wet_texts = {}
    
    stream = open(wet_path, 'rb') if isinstance(wet_path, str) else wet_path
    
    for record in ArchiveIterator(stream, record_types=WarcRecordType.conversion):
        target_uri = record.headers.get('WARC-Target-URI')
        if not target_uri:
            continue
        
        payload = record.reader.read().decode('utf-8', errors='replace')
        wet_texts[target_uri] = payload
        
    return wet_texts


def download_sample_pair() -> Tuple[str, str]:
    """从 Common Crawl 官方下载一对匹配的小体积 WARC 和 WET 文件（若本地不存在）"""
    warc_url = "https://data.commoncrawl.org/crawl-data/CC-MAIN-2024-10/segments/1707947472750.6/warc/CC-MAIN-20240215003554-20240215033554-00000.warc.gz"
    wet_url = "https://data.commoncrawl.org/crawl-data/CC-MAIN-2024-10/segments/1707947472750.6/wet/CC-MAIN-20240215003554-20240215033554-00000.warc.wet.gz"
    
    warc_file = "local-shared-data/CC/example.warc.gz"
    wet_file = "local-shared-data/CC/example.warc.wet.gz"
    
    if not os.path.exists(warc_file):
        print(">>> 正在下载 WARC 示例文件 (大约 50MB)...")
        res = requests.get(warc_url, stream=True)
        with open(warc_file, 'wb') as f:
            for chunk in res.iter_content(chunk_size=1024*1024):
                f.write(chunk)
                
    if not os.path.exists(wet_file):
        print(">>> 正在下载 WET 示例文件 (大约 10MB)...")
        res = requests.get(wet_url, stream=True)
        with open(wet_file, 'wb') as f:
            for chunk in res.iter_content(chunk_size=1024*1024):
                f.write(chunk)
                
    return warc_file, wet_file


def run_comparison(warc_path: str, wet_path: str, max_samples: int = 3):
    """对比解析并输出日志"""
    print("\n--- 开始解析 WARC 文件（使用 Resiliparse 现场提取）---")
    warc_data = read_warc_records(warc_path)
    print(f"成功解析 HTML 网页数: {len(warc_data)}")

    print("\n--- 开始解析 WET 文件（读取官方预提取文本）---")
    wet_data = read_wet_records(wet_path)
    print(f"成功读取 WET 记录数: {len(wet_data)}")

    # 寻找匹配的 URI 进行对比
    common_uris = set(warc_data.keys()).intersection(set(wet_data.keys()))
    print(f"\n找到匹配的网页 URL 数量: {len(common_uris)}")
    print("=" * 70)

    count = 0
    for uri in common_uris:
        resil_text = warc_data[uri].strip()
        wet_text = wet_data[uri].strip()
        
        # 跳过提取为空的内容
        if not resil_text or not wet_text:
            continue

        count += 1
        print(f"\n【样本 {count}】 URL: {uri}")
        print("-" * 70)
        
        print(f"▶ [Resiliparse 提取文本] (长度: {len(resil_text)} 字符)")
        print(resil_text[:350] + ("..." if len(resil_text) > 350 else ""))
        
        print("\n" + "." * 40 + "\n")
        
        print(f"▶ [WET 官方预提取文本] (长度: {len(wet_text)} 字符)")
        print(wet_text[:350] + ("..." if len(wet_text) > 350 else ""))
        
        print("=" * 70)
        
        if count >= max_samples:
            break


if __name__ == '__main__':
    # 自动下载示例文件并运行对比
    warc_f, wet_f = download_sample_pair()
    run_comparison(warc_f, wet_f, max_samples=3)