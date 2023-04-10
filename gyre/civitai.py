import hashlib
import json
import logging
import os
import re
import sqlite3
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from urllib.parse import parse_qs, urlparse
from urllib.request import Request, urlopen

from huggingface_hub.file_download import http_get

from gyre.constants import sd_cache_home

logger = logging.getLogger(__name__)


def sha256sum(filename):
    h = hashlib.sha256()
    b = bytearray(128 * 1024)
    mv = memoryview(b)
    with open(filename, "rb", buffering=0) as f:
        while n := f.readinto(mv):
            h.update(mv[:n])
    return h.hexdigest()


@dataclass
class CivitaiModelReference:
    model_id: int | None = None
    model_version_id: int | None = None
    type: Literal["Model", "Pruned Model"] | None = None


def parse_url(url: str) -> CivitaiModelReference:
    parts = urlparse(url)

    if parts.scheme in {"http", "https"} and parts.netloc == "civitai.com":

        if match := re.match(r"/models/([0-9]+)/", parts.path):
            return CivitaiModelReference(model_id=int(match[1]))

        elif match := re.match(r"/api/download/models/([0-9]+)", parts.path):
            if parts.query:
                params = parse_qs(parts.query)
                if type_param := params.get("type"):
                    return CivitaiModelReference(
                        model_version_id=int(match[1]), type=type_param
                    )

            return CivitaiModelReference(model_version_id=int(match[1]))

    raise ValueError(f"{url} doesn't look like a civitai URL we can understand")


HEADERS = {
    "Host": "civitai.com",
    "User-agent": "gyre.ai/1.0",
    "Accept": "*/*",
}

cache_path = os.path.join(sd_cache_home, "civitai")
os.makedirs(cache_path, exist_ok=True)


@contextmanager
def get_cur():
    db_file = os.path.join(cache_path, "cache.db")
    if not os.path.exists(db_file):
        con = sqlite3.connect(db_file)
        con.row_factory = sqlite3.Row
        cur = con.cursor()
        cur.execute(
            "CREATE TABLE civitai(model_id, model_version_id, timestamp, filestr)"
        )

    else:
        con = sqlite3.connect(db_file)
        con.row_factory = sqlite3.Row
        cur = con.cursor()

    try:
        yield cur
    finally:
        con.commit()
        con.close()


def get_local_candidate(modelref: CivitaiModelReference):
    query = None

    if modelref.model_version_id:
        query = (
            "SELECT * FROM civitai "
            "WHERE model_version_id = :model_version_id "
            "ORDER BY timestamp DESC"
        )
    elif modelref.model_id:
        query = (
            "SELECT * FROM civitai "
            "WHERE model_id = :model_id "
            "ORDER BY timestamp DESC"
        )

    if query:
        with get_cur() as cur:
            return [
                json.loads(row["filestr"])
                for row in cur.execute(query, modelref.__dict__)
            ]


def store_local_candidates(modelref: CivitaiModelReference, files):
    with get_cur() as cur:
        for file in files:
            cur.execute(
                "INSERT INTO civitai (model_id, model_version_id, timestamp, filestr) "
                "VALUES (:model_id, :model_version_id, :timestamp, :filestr)",
                dict(
                    model_id=modelref.model_id,
                    model_version_id=modelref.model_version_id,
                    timestamp=time.time(),
                    filestr=json.dumps(file),
                ),
            )


def get_model(
    modelref: CivitaiModelReference, local_only=False, check_sha=False
) -> str:

    # Find the files data for the model reference
    files = None

    if not local_only:
        if modelref.model_version_id:
            request = Request(
                f"https://civitai.com/api/v1/model-versions/{modelref.model_version_id}",
                headers=HEADERS,
            )

            with urlopen(request) as resp:
                data = json.load(resp)
                # Update the model ref with loaded model id
                modelref.model_id = data["modelId"]
                # Get files
                files = data["files"]

        if modelref.model_id:
            request = Request(
                f"https://civitai.com/api/v1/models/{modelref.model_id}",
                headers=HEADERS,
            )

            with urlopen(request) as resp:
                data = json.load(resp)
                # Update the model ref with loaded model version id
                modelref.model_id = data["modelVersions"][0]["id"]
                # Get files
                files = data["modelVersions"][0]["files"]

    if files:
        store_local_candidates(modelref, files)
    else:
        files = get_local_candidate(modelref)

    if not files:
        if local_only:
            raise ValueError("No local version of CivitAI model available.")
        else:
            raise ValueError("Couldn't get information about model from civitai.com")

    # Search the files for the single file to download

    safetensor, ckpt = None, None

    for file in files:
        if modelref.type and file["type"] != modelref.type:
            continue
        if not safetensor and file["name"].endswith(".safetensors"):
            safetensor = file
        elif not ckpt and file["name"].endswith(".ckpt"):
            ckpt = file

    if safetensor:
        sha256 = safetensor.get("hashes", {}).get("SHA256", None)
        fallback_hashstr = (
            f"{safetensor['name']}:{safetensor['sizeKB']}:{safetensor['metadata']}"
        )
        url = safetensor["downloadUrl"]
        name = "model.safetensors"
    elif ckpt:
        sha256 = ckpt.get("hashes", {}).get("SHA256", None)
        fallback_hashstr = f"{ckpt['name']}:{ckpt['sizeKB']}:{ckpt['metadata']}"
        url = ckpt["downloadUrl"]
        name = "model.ckpt"
    else:
        raise ValueError(
            f"CivitAI model version {modelref.model_version_id} doesn't appear to "
            "have a safetensors or ckpt file"
        )

    fallback_hasher = hashlib.sha256()
    fallback_hasher.update(fallback_hashstr.encode("utf-8"))
    fallback_hash = fallback_hasher.hexdigest()

    def get_path(fullhash: str):
        # Just use the first 24 characters, we're trying to avoid an on-disk collision,
        # not protect against pre-image attacks
        filehash = fullhash[:24].upper()

        base_cache = Path(cache_path) / filehash
        full_name = base_cache / name

        return base_cache, full_name

    full_base, full_name = get_path(sha256) if sha256 else (None, None)
    fallback_base, fallback_name = get_path(fallback_hash)

    # If sha256 named file exists, return it (after potentially checking hash)

    if full_name and full_name.exists():
        real_sha256 = sha256sum(full_name) if check_sha else sha256
        if real_sha256.upper() == sha256.upper():
            return str(full_base)

    # sha256 named file doesn't exist (or doesn't match sha), check fallback

    if fallback_name and fallback_name.exists():
        # If we know the right sha256 hash, rename it & return the renamed version
        if sha256:
            if sha256sum(fallback_name).upper() == sha256.upper():
                logger.info("Moving {fallback_name} to {full_name}")
                full_base.mkdir(exist_ok=True)
                fallback_name.rename(str(full_name))
                try:
                    fallback_base.rmdir()
                except Exception:
                    pass

                return str(full_base)

        else:
            return str(fallback_base)

    # Both sha256 and fallback named files don't exist (or don't match hash).
    # Download if allowed

    if not local_only:
        if full_name:
            download_base, download_name = full_base, full_name
        else:
            download_base, download_name = fallback_base, fallback_name

        temp_path = os.path.join(sd_cache_home, "temp")
        os.makedirs(temp_path, exist_ok=True)

        temp_name = None
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=temp_path, delete=False
        ) as temp_file:
            http_get(url, temp_file)
            temp_name = temp_file.name

        if temp_name:
            os.makedirs(download_base, exist_ok=True)
            os.replace(temp_name, download_name)

        return str(download_base)

    raise ValueError("No local copy of Civitai model, and local_only set")
