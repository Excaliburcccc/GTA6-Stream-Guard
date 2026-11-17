"""Verify and repair the pinned vehicle proposal model used by Stream Guard."""
import hashlib
import os
import shutil
import tempfile
import time
from http.client import IncompleteRead
from urllib.request import Request, urlopen

from police_vehicles import MODEL_PATH, MODEL_SHA256, MODEL_URL


def checksum(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def ensure_model():
    if MODEL_PATH.is_file() and checksum(MODEL_PATH) == MODEL_SHA256:
        print("Police vehicle proposal model verified.", flush=True)
        return

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    request = Request(MODEL_URL, headers={"User-Agent": "GTA6-Stream-Guard"})
    for attempt in range(3):
        with tempfile.NamedTemporaryFile(
            dir=MODEL_PATH.parent, prefix=".patrol-", suffix=".onnx", delete=False
        ) as dst:
            temporary = MODEL_PATH.parent / os.path.basename(dst.name)
        try:
            print("Downloading the police vehicle proposal model (4 MB) ...", flush=True)
            with urlopen(request, timeout=45) as src, temporary.open("wb") as dst:
                shutil.copyfileobj(src, dst, length=1024 * 1024)
            if checksum(temporary) != MODEL_SHA256:
                raise OSError("SHA256 mismatch for the police vehicle proposal model")
            os.replace(temporary, MODEL_PATH)
            print("Police vehicle proposal model downloaded and verified.", flush=True)
            return
        except (OSError, IncompleteRead):
            if attempt == 2:
                raise
            time.sleep(1 + attempt)
        finally:
            temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    ensure_model()
