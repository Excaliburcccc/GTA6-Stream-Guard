"""Download the pinned scene classifier. Media frames are never uploaded."""
import argparse
import hashlib
import os
from http.client import IncompleteRead
from pathlib import Path
import shutil
import tempfile
import time
from urllib.request import urlopen

from scene_guard import DEFAULT_MODEL_PATH, MODEL_ID, MODEL_REVISION


FILES = {
    "config.json": "b575ef3c36f2a057fa19e221650105052d61cc9c1a972ec15019c6261ec98770",
    "merges.txt": "f526393189112391ce6f9795d4695f704121ce452c3aad1f5335cc41337eba85",
    "preprocessor_config.json": "910e70b3956ac9879ebc90b22fb3bc8a75b6a0677814500101a4c072bd7857bd",
    "pytorch_model.bin": "a63082132ba4f97a80bea76823f544493bffa8082296d62d71581a4feff1576f",
    "special_tokens_map.json": "f8c0d6c39aee3f8431078ef6646567b0aba7f2246e9c54b8b99d55c22b707cbf",
    "tokenizer.json": "b556ac8c99757ffb677208af34bc8c6721572114111a6e0aaf5fa69ff0b8d842",
    "tokenizer_config.json": "34b7336e4bee12e0a9730eaf5189f582ef3c3eea5027f65730e5717256755aad",
    "vocab.json": "5047b556ce86ccaf6aa22b3ffccfc52d391ea4accdab9c2f2407da5b742d4363",
}


def checksum(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def download(path, expected_hash):
    if path.is_file() and checksum(path) == expected_hash:
        return
    url = f"https://huggingface.co/{MODEL_ID}/resolve/{MODEL_REVISION}/{path.name}"
    for attempt in range(3):
        fd, temp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}-")
        try:
            print(f"Downloading {path.name} ...", flush=True)
            with os.fdopen(fd, "wb") as target, urlopen(url, timeout=90) as source:
                shutil.copyfileobj(source, target, length=1024 * 1024)
            if checksum(Path(temp)) != expected_hash:
                raise OSError(f"SHA256 mismatch: {path.name}")
            os.replace(temp, path)
            return
        except (OSError, IncompleteRead):
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)
        finally:
            Path(temp).unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description="下载并校验本地场景模型（约 610 MB）")
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for filename, expected_hash in FILES.items():
        download(args.output / filename, expected_hash)
    # Load on CPU as an integrity/compatibility check before reporting success.
    from scene_guard import SceneGuard

    guard = SceneGuard(device="cpu", model_path=args.output)
    if not guard.available:
        parser.exit(1, f"场景模型未能加载：{guard.error}\n")
    print(f"场景模型已下载并通过 CPU 加载检查：{args.output.resolve()}")


if __name__ == "__main__":
    main()
