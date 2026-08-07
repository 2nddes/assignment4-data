import gzip
import shutil
import tempfile
import time
from functools import cached_property, lru_cache
from io import BytesIO
from pathlib import Path

from collections.abc import Callable
import fasttext
import modal
import polars as pl
from warcio.archiveiterator import ArchiveIterator
from warcio.warcwriter import WARCWriter

from cs336_data.common import get_shared_assets_path
from cs336_data.download_utils import commoncrawl_url, download_file
from cs336_data.modal_utils import VOLUME_MOUNTS, app, build_image
from furu import Furu


@lru_cache(maxsize=1)
def _load_lid_model():
    model_path = get_shared_assets_path() / "classifiers" / "lid.176.bin"
    return fasttext.load_model(str(model_path))


def _is_english(text: str) -> bool:
    labels, probabilities = _load_lid_model().predict(text.replace("\n", " ").strip(), k=1)
    return labels[0] == "__label__en" and float(probabilities[0]) >= 0.7


class _EnglishWetFile(Furu[Path]):
    chunk_urls: tuple[str, ...]

    def _create(self) -> Path:
        output_path = self.data_dir / "data.warc.wet.gz"

        print("Loading English language identifier", flush=True)
        is_english: Callable[[str], bool] = _is_english

        total_text = 0
        skipped_text = 0
        print(f"Processing WET chunk ({len(self.chunk_urls)} files)", flush=True)

        with tempfile.NamedTemporaryFile(
            delete=False,
            dir=tempfile.gettempdir(),
            suffix=f".{output_path.name}",
        ) as temp_output_file:
            temp_output_path = Path(temp_output_file.name)

        with gzip.open(temp_output_path, "wb") as output_stream:
            writer = WARCWriter(output_stream, gzip=False)
            for wet_url in self.chunk_urls:
                local_wet_path = Path(tempfile.gettempdir()) / wet_url.split("/")[-1]
                if not local_wet_path.exists():
                    print(f"Downloading {wet_url} to {local_wet_path}", flush=True)
                    download_file(wet_url, local_wet_path, label=local_wet_path.name)
                else:
                    print(f"Using cached WET file {local_wet_path}", flush=True)
                with gzip.open(local_wet_path, "rb") as input_stream:
                    for rec in ArchiveIterator(input_stream):
                        if rec.rec_type != "conversion":
                            writer.write_record(rec)
                            continue
                        payload = rec.content_stream().read()
                        text = payload.decode("utf-8", errors="replace")
                        total_text += len(text)

                        if is_english(text):
                            rec.raw_stream = BytesIO(payload)
                            writer.write_record(rec)
                        else:
                            skipped_text += len(text)
        shutil.copy2(temp_output_path, output_path)
        temp_output_path.unlink(missing_ok=True)

        print(
            f"Finished WET chunk: wrote {output_path}, "
            f"kept {100 * (total_text - skipped_text) / total_text if total_text else 0:.2f}% of text",
            flush=True,
        )
        return output_path

    @cached_property
    def storage_root(self) -> Path:
        return get_shared_assets_path() / "furu"


@app.function(image=build_image(), volumes=VOLUME_MOUNTS, timeout=60 * 60 * 12, max_containers=128)
def make_wet_file_on_modal(wet_file: _EnglishWetFile) -> Path:
    return wet_file.load_or_create()


class EnglishWetFiles(Furu[list[Path]]):
    n_files: int = 2500
    group_size: int = 4
    shuffle_seed: int = 336
    crawl_id: str = "CC-MAIN-2026-17"

    def _create(self) -> list[Path]:
        n_files = self.n_files - (self.n_files % self.group_size)
        if n_files != self.n_files:
            print(
                f"[wet] rounded --wet-files {self.n_files} down to {n_files} "
                f"(must be a multiple of group_size {self.group_size})",
                flush=True,
            )
        wet_paths_url = commoncrawl_url("crawl-data", self.crawl_id, "wet.paths.gz")
        wet_paths_file = get_shared_assets_path() / "metadata" / f"{self.crawl_id}-wet.paths.gz"
        print(f"[wet] loading WET paths from {wet_paths_url}", flush=True)
        download_file(wet_paths_url, wet_paths_file, label=wet_paths_file.name)

        all_wet_urls = (
            pl.read_csv(
                wet_paths_file,
                has_header=False,
                new_columns=["wet_path"],
            )
            .with_columns(pl.col("wet_path").map_elements(commoncrawl_url, return_dtype=pl.String))
            .sample(fraction=1.0, shuffle=True, seed=self.shuffle_seed)["wet_path"]
            .to_list()
        )
        # Deterministic prefix selection: rerunning with a larger `n_files` reuses
        # the chunks already completed by smaller runs.
        wet_urls = all_wet_urls[:n_files]

        print(f"[wet] selected {len(wet_urls)} of {len(all_wet_urls)} WET files for crawl {self.crawl_id}", flush=True)

        wet_files: list[_EnglishWetFile] = []
        for chunk_idx in range(0, len(wet_urls), self.group_size):
            chunk_urls = tuple(wet_urls[chunk_idx : chunk_idx + self.group_size])
            wet_files.append(_EnglishWetFile(chunk_urls=chunk_urls))

        completed = sum(wf.is_completed() for wf in wet_files)
        print(f"[wet] {completed}/{len(wet_files)} chunks already completed", flush=True)

        wet_data_paths: list[Path] = []
        if modal.is_local():
            print("[wet] downloading wet files locally", flush=True)
            t_start = time.monotonic()
            for wet_file_idx, wet_file in enumerate(wet_files, start=1):
                if wet_file.is_completed():
                    path = wet_file.try_load()
                    print(f"[wet] {wet_file_idx}/{len(wet_files)} cached -> {path}", flush=True)
                else:
                    t0 = time.monotonic()
                    path = wet_file.load_or_create()
                    print(
                        f"[wet] {wet_file_idx}/{len(wet_files)} done in {time.monotonic() - t0:.1f}s -> {path}",
                        flush=True,
                    )
                wet_data_paths.append(path)
            print(f"[wet] finished {len(wet_data_paths)} chunks in {time.monotonic() - t_start:.1f}s", flush=True)
        else:
            print("[wet] downloading wet files on remote", flush=True)

            wet_data_paths = list(make_wet_file_on_modal.map(wet_files))
            print(f"[wet] completed {len(wet_data_paths)} remote WET chunks", flush=True)

            repo_path = get_shared_assets_path() / "english-wet-data"
            repo_path.mkdir(exist_ok=False)
            print(f"[wet] linking remote WET outputs into {repo_path}", flush=True)

            source_link = repo_path / ".source"
            if source_link.exists() or source_link.is_symlink():
                print(f"[wet] replacing existing source link {source_link}", flush=True)
                source_link.unlink()
            source_link.symlink_to(self.data_dir)
            print(f"[wet] linked source data directory {source_link} -> {self.data_dir}", flush=True)

            for wet_data_idx, wet_data_path in enumerate(wet_data_paths):
                link_path = repo_path / f"{wet_data_idx:05d}-{wet_data_path.name}"
                if link_path.exists() or link_path.is_symlink():
                    print(f"[wet] replacing existing WET chunk link {link_path}", flush=True)
                    link_path.unlink()
                link_path.symlink_to(wet_data_path)
                print(f"[wet] linked WET chunk {wet_data_idx}: {link_path} -> {wet_data_path}", flush=True)

        print(f"[wet] finished creating {len(wet_data_paths)} English WET files", flush=True)
        return wet_data_paths

    @cached_property
    def storage_root(self) -> Path:
        return get_shared_assets_path() / "furu"
