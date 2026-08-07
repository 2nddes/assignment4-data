import argparse
import bz2
import gzip
import re
import shutil
import tempfile
import time
from pathlib import Path

import modal

from cs336_data.common import get_shared_assets_path
from cs336_data.download_utils import (
    commoncrawl_url,
    download_file,
    fasttext_lid_url,
    human_size,
    huggingface_resolve_url,
    urlopen_with_retries,
    wikimedia_url,
)
from cs336_data.modal_utils import VOLUME_MOUNTS, app, build_image
from cs336_data.wet_files import EnglishWetFiles

""" RUN
$env:CS336_HF_ENDPOINT = "https://hf-mirror.com"
$env:HF_ENDPOINT = "https://hf-mirror.com"
$env:CS336_DOWNLOAD_RETRIES = "5"
$env:CS336_DOWNLOAD_TIMEOUT = "120"

# Small first run (resume-safe; later runs can raise the limits and reuse completed work):
uv run scripts/download_data.py --max-wiki-shards 5 --wet-files 100

# Full run:
uv run scripts/download_data.py

# Only the offline files (classifiers, example WARC/WET, paloma validation data):
uv run scripts/download_data.py --offline-only
"""

DUMP_DATE = "20260501"
URL_RE = re.compile(
    r"\b(?:https?|telnet|gopher|file|wais|ftp):[\w/#~:.?+=&%@!\-.:?\\-]+?(?=[.:?\-]*(?:[^\w/#~:.?+=&%@!\-.:?\-]|$))"
)


def _iter_wiki_urls(dump: Path):
    with bz2.open(dump, "rt", errors="ignore") as f:
        for line in f:
            if refs := re.search("&lt;ref&gt(.*)&lt;/ref&gt;", line):
                yield from URL_RE.findall(refs.group(0))


@app.function(image=build_image(), volumes=VOLUME_MOUNTS, timeout=60 * 60 * 12, max_containers=128)
def extract_wiki_urls(shard: str) -> list[str]:
    tmp_dir = Path(tempfile.gettempdir()) / "wiki"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    dump = tmp_dir / shard

    print(f"[wiki] downloading {shard}", flush=True)
    download_file(wikimedia_url("enwiki", DUMP_DATE, shard), dump, label=shard)
    urls = list(_iter_wiki_urls(dump))
    dump.unlink(missing_ok=True)
    return urls


def extract_wiki_shard_local(shard: str, out: Path) -> int:
    """Download one Wikipedia shard and write its extracted URLs atomically to `out`."""
    tmp_dir = Path(tempfile.gettempdir()) / "wiki"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    dump = tmp_dir / shard

    print(f"[wiki] downloading {shard}", flush=True)
    download_file(wikimedia_url("enwiki", DUMP_DATE, shard), dump, label=shard)
    part = out.with_name(out.name + ".part")
    count = 0
    with part.open("w", encoding="utf-8") as f:
        for url in _iter_wiki_urls(dump):
            f.write(url + "\n")
            count += 1
    dump.unlink(missing_ok=True)
    part.replace(out)
    return count


def get_wiki_shards() -> list[str]:
    base_url = wikimedia_url("enwiki", DUMP_DATE)
    with urlopen_with_retries(base_url) as response:
        html = response.read().decode()
    return sorted(
        set(re.findall(rf"enwiki-{DUMP_DATE}-pages-articles-multistream[0-9]+\.xml-p[0-9]+p[0-9]+\.bz2", html))
    )


def extract_wiki_urls_to_dir(shards: list[str], urls_dir: Path) -> None:
    """Extract URLs for `shards` into per-shard files, skipping shards already done."""
    urls_dir.mkdir(parents=True, exist_ok=True)
    pending = [s for s in shards if not (urls_dir / f"{s}.txt").exists()]
    done = len(shards) - len(pending)
    if done:
        print(f"[wiki] {done}/{len(shards)} shards already extracted; resuming", flush=True)
    if not pending:
        return

    if modal.is_local():
        for i, shard in enumerate(pending, start=1):
            out = urls_dir / f"{shard}.txt"
            t0 = time.monotonic()
            count = extract_wiki_shard_local(shard, out)
            print(
                f"[wiki] {done + i}/{len(shards)} {shard}: {count:,} URLs in {time.monotonic() - t0:.1f}s",
                flush=True,
            )
    else:
        for i, (shard, urls) in enumerate(zip(pending, extract_wiki_urls.map(pending)), start=1):
            out = urls_dir / f"{shard}.txt"
            part = out.with_name(out.name + ".part")
            with part.open("w", encoding="utf-8") as f:
                for url in urls:
                    f.write(url + "\n")
            part.replace(out)
            print(f"[wiki] {done + i}/{len(shards)} {shard}: {len(urls):,} URLs", flush=True)


def build_wiki_output(urls_dir: Path, wiki_out: Path) -> int | None:
    """Combine all per-shard URL files into one gzipped file (rebuild only when shards changed)."""
    shard_files = sorted(urls_dir.glob("*.txt"))
    if not shard_files:
        print("[wiki] no extracted shard files yet; skipping combined output", flush=True)
        return None

    marker = wiki_out.with_name(wiki_out.name + ".shards")
    expected = "\n".join(sf.name for sf in shard_files)
    if wiki_out.exists() and marker.exists() and marker.read_text(encoding="utf-8") == expected:
        print(f"[wiki] combined output up to date ({len(shard_files)} shards): {wiki_out}", flush=True)
        return None

    tmp_out = Path(tempfile.gettempdir()) / wiki_out.name
    tmp_out.unlink(missing_ok=True)
    total = 0
    with gzip.open(tmp_out, "wt", encoding="utf-8") as f:
        for sf in shard_files:
            with sf.open("r", encoding="utf-8") as inf:
                for line in inf:
                    f.write(line)
                    total += 1
    shutil.copy2(tmp_out, wiki_out)
    tmp_out.unlink(missing_ok=True)
    marker.write_text(expected, encoding="utf-8")
    print(f"[wiki] wrote {wiki_out} ({total:,} URLs from {len(shard_files)} shards)", flush=True)
    return total


def download_offline_files(*, root_path: Path) -> None:
    paloma_out = root_path / "tokenized_paloma_c4_100_domains_validation.bin"
    download_file(
        huggingface_resolve_url(
            "brunborg/cs336-a4",
            "tokenized_paloma_c4_100_domains_validation.bin",
            repo_type="dataset",
        ),
        paloma_out,
        label=paloma_out.name,
    )

    cc = root_path / "CC"
    cc.mkdir(parents=True, exist_ok=True)
    for kind, out_name in [("warc", "example.warc.gz"), ("wet", "example.warc.wet.gz")]:
        out = cc / out_name
        if out.exists() and out.stat().st_size > 0:
            print(f"[download] using cached {out} ({human_size(out.stat().st_size)})", flush=True)
            continue

        paths_file = root_path / "metadata" / f"CC-MAIN-2026-12-{kind}.paths.gz"
        download_file(
            commoncrawl_url("crawl-data", "CC-MAIN-2026-12", f"{kind}.paths.gz"),
            paths_file,
            label=paths_file.name,
        )
        first_path = gzip.decompress(paths_file.read_bytes()).decode().splitlines()[0]
        download_file(commoncrawl_url(first_path), out, label=out_name)

    for rel_path, url in [
        (
            "classifiers/lid.176.bin",
            fasttext_lid_url(),
        ),
        (
            "classifiers/dolma_fasttext_hatespeech_jigsaw_model.bin",
            huggingface_resolve_url("allenai/dolma-jigsaw-fasttext-bigrams-hatespeech", "model.bin"),
        ),
        (
            "classifiers/dolma_fasttext_nsfw_jigsaw_model.bin",
            huggingface_resolve_url("allenai/dolma-jigsaw-fasttext-bigrams-nsfw", "model.bin"),
        ),
    ]:
        out = root_path / rel_path
        download_file(url, out, label=rel_path)


def print_summary(*, root_path: Path, wiki_out: Path | None, wet_file_paths: list[Path]) -> None:
    print("=== summary ===", flush=True)
    offline_rels = [
        "tokenized_paloma_c4_100_domains_validation.bin",
        "CC/example.warc.gz",
        "CC/example.warc.wet.gz",
        "classifiers/lid.176.bin",
        "classifiers/dolma_fasttext_hatespeech_jigsaw_model.bin",
        "classifiers/dolma_fasttext_nsfw_jigsaw_model.bin",
    ]
    for rel in offline_rels:
        p = root_path / rel
        if p.exists():
            print(f"  {rel}: {human_size(p.stat().st_size)}", flush=True)
    if wiki_out is not None and wiki_out.exists():
        print(f"  wiki/{wiki_out.name}: {human_size(wiki_out.stat().st_size)}", flush=True)
    if wet_file_paths:
        total_bytes = sum(p.stat().st_size for p in wet_file_paths if p.exists())
        print(f"  english-wet chunks: {len(wet_file_paths)} files, {human_size(total_bytes)}", flush=True)


@app.function(image=build_image(), volumes=VOLUME_MOUNTS, timeout=60 * 60 * 12)
def main(offline_only: bool = False, max_wiki_shards: int | None = None, wet_files: int = 2500):
    root_path = get_shared_assets_path()
    t_start = time.monotonic()
    print(f"=== data root: {root_path} ===", flush=True)

    download_offline_files(root_path=root_path)
    if offline_only:
        print(f"[done] offline files ready in {time.monotonic() - t_start:.1f}s", flush=True)
        return

    wiki_out: Path | None = None
    if max_wiki_shards == 0:
        print("[wiki] skipping wiki URL extraction (--max-wiki-shards 0)", flush=True)
    else:
        shards = get_wiki_shards()
        if max_wiki_shards is not None:
            shards = shards[:max_wiki_shards]
        print(
            f"[wiki] {len(shards)} shards selected (max_wiki_shards={max_wiki_shards if max_wiki_shards is not None else 'all'})",
            flush=True,
        )
        wiki_out = root_path / "wiki" / f"enwiki-{DUMP_DATE}-extracted_urls.txt.gz"
        if shards:
            urls_dir = wiki_out.parent / "urls"
            extract_wiki_urls_to_dir(shards, urls_dir)
            build_wiki_output(urls_dir, wiki_out)

    if wet_files <= 0:
        print("[wet] skipping WET processing (--wet-files 0)", flush=True)
        wet_file_paths: list[Path] = []
    else:
        english_wet_files = EnglishWetFiles(n_files=wet_files)
        wet_file_paths = english_wet_files.load_or_create()
        print(f"[wet] {len(wet_file_paths)} chunks ready", flush=True)

    print_summary(root_path=root_path, wiki_out=wiki_out, wet_file_paths=wet_file_paths)
    print(f"[done] total time {time.monotonic() - t_start:.1f}s", flush=True)


@app.local_entrypoint()
def modal_main(offline_only: bool = False, max_wiki_shards: int | None = None, wet_files: int = 2500):
    main.remote(offline_only=offline_only, max_wiki_shards=max_wiki_shards, wet_files=wet_files)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and prepare data for CS336 Assignment 4.")
    parser.add_argument(
        "--offline-only",
        action="store_true",
        help="Only download files needed for running the assignment offline; skip full WET/wiki data creation.",
    )
    parser.add_argument(
        "--max-wiki-shards",
        type=int,
        default=None,
        help="Only extract URLs from the first N Wikipedia shards (0 skips wiki entirely; default: all).",
    )
    parser.add_argument(
        "--wet-files",
        type=int,
        default=2500,
        help="Number of WET files to download and process (0 skips WET; default: 2500). "
        "Rerunning with a larger value resumes and extends previously completed chunks.",
    )
    args = parser.parse_args()
    main.local(
        offline_only=args.offline_only,
        max_wiki_shards=args.max_wiki_shards,
        wet_files=args.wet_files,
    )
