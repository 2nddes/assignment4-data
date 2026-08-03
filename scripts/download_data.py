import bz2
import gzip
import re
import shutil
from pathlib import Path

import modal

from cs336_data.common import get_shared_assets_path
from cs336_data.download_utils import (
    commoncrawl_url,
    download_file,
    fasttext_lid_url,
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
uv run scripts/download_data.py --offline-only
"""


@app.function(image=build_image(), volumes=VOLUME_MOUNTS, timeout=60 * 60 * 12, max_containers=128)
def extract_wiki_urls(shard: str) -> list[str]:
    dump_date = "20260501"
    tmp_dir = Path("/tmp/wiki")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    dump = tmp_dir / shard
    url_re = re.compile(
        r"\b(?:https?|telnet|gopher|file|wais|ftp):[\w/#~:.?+=&%@!\-.:?\\-]+?(?=[.:?\-]*(?:[^\w/#~:.?+=&%@!\-.:?\-]|$))"
    )

    print(f"[wiki] downloading {shard}", flush=True)
    download_file(wikimedia_url("enwiki", dump_date, shard), dump, label=shard)
    urls = []
    with bz2.open(dump, "rt", errors="ignore") as f:
        for line in f:
            if refs := re.search("&lt;ref&gt(.*)&lt;/ref&gt;", line):
                urls.extend(url_re.findall(refs.group(0)))
    dump.unlink(missing_ok=True)
    return urls


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
            print(f"[download] using cached {out}", flush=True)
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


@app.function(image=build_image(), volumes=VOLUME_MOUNTS, timeout=60 * 60 * 12)
def main(offline_only: bool = False):
    root_path = get_shared_assets_path()
    download_offline_files(root_path=root_path)
    if offline_only:
        return

    dump_date = "20260501"
    base_url = wikimedia_url("enwiki", dump_date)
    with urlopen_with_retries(base_url) as response:
        html = response.read().decode()
    shards = sorted(
        set(re.findall(rf"enwiki-{dump_date}-pages-articles-multistream[0-9]+\.xml-p[0-9]+p[0-9]+\.bz2", html))
    )
    wiki_out = root_path / "wiki/enwiki-20260501-extracted_urls.txt.gz"
    if not wiki_out.exists():
        wiki_out.parent.mkdir(parents=True, exist_ok=True)
        tmp_out = Path("/tmp") / wiki_out.name
        tmp_out.unlink(missing_ok=True)
        print(f"[wiki] extracting {len(shards)} shards", flush=True)
        with gzip.open(tmp_out, "wt") as f:
            for urls in (
                [extract_wiki_urls.local(shard) for shard in shards]
                if modal.is_local()
                else extract_wiki_urls.map(shards)
            ):
                for url in urls:
                    f.write(url + "\n")
        shutil.copy2(tmp_out, wiki_out)
        tmp_out.unlink(missing_ok=True)
        print(f"[wiki] wrote {wiki_out}", flush=True)

    english_wet_files = EnglishWetFiles()
    wet_file_paths = english_wet_files.load_or_create()
    print(f"downloaded {len(wet_file_paths)} including {wet_file_paths[0]=}")


@app.local_entrypoint()
def modal_main(offline_only: bool = False):
    main.remote(offline_only=offline_only)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--offline-only",
        action="store_true",
        help="Only download files needed for running the assignment offline; skip full WET/wiki data creation.",
    )
    args = parser.parse_args()
    main.local(offline_only=args.offline_only)
