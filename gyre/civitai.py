import json
import os
import re
import sqlite3
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Literal
from urllib.parse import parse_qs, urlparse
from urllib.request import Request, urlopen

from huggingface_hub.file_download import http_get

from gyre.constants import sd_cache_home


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


def get_model(modelref: CivitaiModelReference, local_only=False):

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
        sha256 = safetensor["hashes"]["SHA256"]
        url = safetensor["downloadUrl"]
        name = "model.safetensors"
    elif ckpt:
        sha256 = ckpt["hashes"]["SHA256"]
        url = ckpt["downloadUrl"]
        name = "model.ckpt"
    else:
        raise ValueError(
            f"CivitAI model version {modelref.model_version_id} doesn't appear to "
            "have a safetensors or ckpt file"
        )

    # Just use the first 24 characters, we're trying to avoid an on-disk collision,
    # not potect against pre-image attacks

    filehash = sha256[:24]

    # Download the file if not already available

    base_cache = os.path.join(cache_path, filehash)
    full_name = os.path.join(base_cache, name)

    if not os.path.exists(full_name) and not local_only:
        temp_path = os.path.join(sd_cache_home, "temp")
        os.makedirs(temp_path, exist_ok=True)

        temp_name = None
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=temp_path, delete=False
        ) as temp_file:
            http_get(url, temp_file)
            temp_name = temp_file.name

        if temp_name:
            os.makedirs(base_cache, exist_ok=True)
            os.replace(temp_name, full_name)

    if os.path.exists(full_name):
        return base_cache
    else:
        raise ValueError("Couldn't download Civitai model")
