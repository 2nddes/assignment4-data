import os
import shutil
import time
import urllib.error
import urllib.request
from pathlib import Path


USER_AGENT = "cs336-data-downloader/1.0"
DEFAULT_DOWNLOAD_RETRIES = 3
DEFAULT_DOWNLOAD_TIMEOUT_SECONDS = 60


def get_env_url(name: str, default: str) -> str:
    value = os.environ.get(name, default).strip()
    if not value:
        value = default
    return value.rstrip("/")


def get_download_retries() -> int:
    return int(os.environ.get("CS336_DOWNLOAD_RETRIES", DEFAULT_DOWNLOAD_RETRIES))


def get_download_timeout_seconds() -> int:
    return int(os.environ.get("CS336_DOWNLOAD_TIMEOUT", DEFAULT_DOWNLOAD_TIMEOUT_SECONDS))


def join_url(base_url: str, *parts: str) -> str:
    suffix = "/".join(str(part).strip("/") for part in parts if str(part).strip("/"))
    return f"{base_url.rstrip('/')}/{suffix}" if suffix else base_url.rstrip("/")


def huggingface_base_url() -> str:
    return get_env_url("CS336_HF_ENDPOINT", os.environ.get("HF_ENDPOINT", "https://huggingface.co"))


def commoncrawl_base_url() -> str:
    return get_env_url("CS336_COMMONCRAWL_BASE_URL", "https://data.commoncrawl.org")


def wikimedia_base_url() -> str:
    return get_env_url("CS336_WIKIMEDIA_BASE_URL", "https://dumps.wikimedia.org")


def fasttext_base_url() -> str:
    return get_env_url("CS336_FASTTEXT_BASE_URL", "https://dl.fbaipublicfiles.com")


def huggingface_resolve_url(repo_id: str, filename: str, *, repo_type: str = "model", revision: str = "main") -> str:
    repo_type_prefix = {
        "model": "",
        "dataset": "datasets",
        "space": "spaces",
    }[repo_type]
    return join_url(huggingface_base_url(), repo_type_prefix, repo_id, "resolve", revision, filename)


def commoncrawl_url(*parts: str) -> str:
    return join_url(commoncrawl_base_url(), *parts)


def wikimedia_url(*parts: str) -> str:
    return join_url(wikimedia_base_url(), *parts)


def fasttext_lid_url() -> str:
    return join_url(fasttext_base_url(), "fasttext", "supervised-models", "lid.176.bin")


def urlopen_with_retries(url: str):
    last_error: Exception | None = None
    retries = get_download_retries()
    timeout = get_download_timeout_seconds()
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            return urllib.request.urlopen(request, timeout=timeout)
        except Exception as error:
            last_error = error
            if attempt + 1 == retries:
                break
            time.sleep(min(2**attempt, 10))
    assert last_error is not None
    raise last_error


def download_file(url: str, output_path: Path, *, label: str | None = None) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.stat().st_size > 0:
        print(f"[download] using cached {output_path}", flush=True)
        return output_path

    part_path = output_path.with_name(f"{output_path.name}.part")
    retries = get_download_retries()
    timeout = get_download_timeout_seconds()
    last_error: Exception | None = None

    for attempt in range(retries):
        resume_from = part_path.stat().st_size if part_path.exists() else 0
        headers = {"User-Agent": USER_AGENT}
        if resume_from > 0:
            headers["Range"] = f"bytes={resume_from}-"

        try:
            name = label or output_path.name
            action = "resuming" if resume_from else "downloading"
            print(f"[download] {action} {name} from {url}", flush=True)
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=timeout) as response:
                status = getattr(response, "status", response.getcode())
                can_append = resume_from > 0 and status == 206
                if resume_from > 0 and not can_append:
                    print(f"[download] server did not resume {name}; restarting", flush=True)
                mode = "ab" if can_append else "wb"
                with part_path.open(mode) as output_file:
                    shutil.copyfileobj(response, output_file)

            if not part_path.exists() or part_path.stat().st_size == 0:
                raise RuntimeError(f"download produced an empty file: {url}")
            part_path.replace(output_path)
            return output_path
        except urllib.error.HTTPError as error:
            last_error = error
            if error.code == 416 and part_path.exists():
                print(f"[download] cached partial file is not resumable; restarting {label or output_path.name}", flush=True)
                part_path.unlink()
        except Exception as error:
            last_error = error

        if attempt + 1 < retries:
            time.sleep(min(2**attempt, 10))

    assert last_error is not None
    raise last_error
