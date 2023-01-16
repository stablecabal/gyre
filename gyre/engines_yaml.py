import glob
import hashlib
import json
import os
import shutil
import subprocess

import yaml

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIST_CONFIG_PATH = os.path.join(BASE_PATH, "gyre/config")
GENHASH_PATH = os.path.join(DIST_CONFIG_PATH, "dist_hashes.json")

# All the paths config has been distributed in historically
DIST_CONFIG_PATHS = ["engines.yaml", "sdgrpcserver/config", "gyre/config"]


# This method from https://github.com/pydantic/pydantic/blob/main/pydantic/_internal/_utils.py#L115
def deep_update(mapping, *updating_mappings):
    updated_mapping = mapping.copy()
    for updating_mapping in updating_mappings:
        for k, v in updating_mapping.items():
            if (
                k in updated_mapping
                and isinstance(updated_mapping[k], dict)
                and isinstance(v, dict)
            ):
                updated_mapping[k] = deep_update(updated_mapping[k], v)
            else:
                updated_mapping[k] = v
    return updated_mapping


def git_object_hash(bs: bytes):
    hasher = hashlib.sha1()
    hasher.update(b"blob ")
    hasher.update(bytes(str(len(bs)), "utf-8"))
    hasher.update(b"\0")
    hasher.update(bs)
    return hasher.hexdigest()


class Loader(yaml.SafeLoader):
    def __init__(self, stream):
        self._root = os.path.split(stream.name)[0]
        super(Loader, self).__init__(stream)

    def subfile(self, tag, node):
        res = []

        for filename in glob.glob(
            os.path.join(self._root, self.construct_scalar(node))
        ):
            with open(filename, "r") as f:
                data = yaml.load(f, Loader)
                if isinstance(data, list):
                    res.extend(data)
                elif data is None:
                    pass
                else:
                    res.append(data)

        return {"_subfile": tag, "res": res}

    def include(self, node):
        return self.subfile("include", node)

    def merge(self, node):
        return self.subfile("merge", node)


Loader.add_constructor("!merge", Loader.merge)
Loader.add_constructor("!include", Loader.include)


class EnginesYaml:
    def __init__(self, stream):
        # Flatten out the data
        self.engines = {}
        self.models = {}

        # Load the raw yaml data
        data = yaml.load(stream, Loader=Loader)
        self.include_data(data)

    def handle_subfile(self, item):
        if item["_subfile"] == "include":
            self.include_data(item["res"])
        elif item["_subfile"] == "merge":
            self.merge_data(item["res"])
        else:
            raise RuntimeError(f"Unknown subfile inclusion method {item['_subfile']}")

    def include_data(self, data):
        for item in data:
            if "id" in item:
                id = item["id"]
                if id in self.engines:
                    print(f"Warning: overwriting engine with duplicate ID {id}.")
                self.engines[id] = item

            elif "model_id" in item:
                model_id = item["model_id"]
                if model_id in self.models:
                    print(f"Warning: overwriting model with duplicate ID {model_id}.")
                self.models[model_id] = item

            elif "_subfile" in item:
                self.handle_subfile(item)

            else:
                raise RuntimeError(
                    "Item in engines config isn't an engine, model or subfile"
                )

    def merge_data(self, data):
        for item in data:
            if "id" in item:
                id = item["id"]
                if id not in self.engines:
                    self.engines[id] = item
                else:
                    self.engines[id] = deep_update(self.engines[id], item)

            elif "model_id" in item:
                model_id = item["model_id"]
                if model_id not in self.models:
                    self.models[model_id] = item
                else:
                    self.models[model_id] = deep_update(self.models[model_id], item)

            elif "_subfile" in item:
                self.handle_subfile(item)

            else:
                raise RuntimeError(
                    "Item in engines config isn't an engine, model or subfile"
                )

    @classmethod
    def load(cls, stream):
        res = cls(stream)
        return list(res.models.values()) + list(res.engines.values())

    @classmethod
    def check_and_update(cls, config_path):

        with open(GENHASH_PATH, "r") as f:
            dist_hashes = json.load(f)
            for path, hashes in dist_hashes.items():

                update = True

                # Check the current config file to see if it matches a distribution hash
                # (if not we ingore it, as it's user edited)

                current_hash = None
                current_path = os.path.join(config_path, path)

                if os.path.isfile(current_path):
                    current_hash = git_object_hash(open(current_path, "rb").read())
                    if current_hash not in hashes:
                        update = False

                # Now check the distribution config file to see if it
                # [a] exists (otherwise it's a since deleted file and needs removing)
                # [b] doesn't match current_hash (in which case we update it)

                dist_path = os.path.join(DIST_CONFIG_PATH, path)

                if os.path.isfile(dist_path):
                    dist_hash = git_object_hash(open(dist_path, "rb").read())
                    if current_hash != dist_hash:
                        if not update:
                            print(
                                f"Config file {path} has been edited, and won't be changed."
                            )
                        else:
                            print("Updating config file", path)
                            os.makedirs(os.path.dirname(current_path), exist_ok=True)
                            shutil.copyfile(dist_path, current_path)
                elif update:
                    print(
                        "Config file", path, "appears to be obsolete and can be removed"
                    )


def gen_hashes(outpath):
    collected = {}

    for path in DIST_CONFIG_PATHS:
        res = subprocess.run(
            ["git", "log", "--name-only", "--oneline", "--", path],
            cwd=BASE_PATH,
            capture_output=True,
            text=True,
        )

        hash = None
        files = []

        for line in res.stdout.splitlines():
            if line.startswith(path):
                if line.endswith(".yaml"):
                    files.append(line)
            else:
                if hash and files:
                    collected[hash] = files
                files = []
                hash = line.split(" ")[0]

        if hash and files:
            collected[hash] = files

    hashes = {}

    for hash, files in collected.items():
        for file in files:
            res = subprocess.run(
                ["git", "rev-parse", f"{hash}:{file}"],
                cwd=BASE_PATH,
                capture_output=True,
                text=True,
            )

            basefile = file
            for path in DIST_CONFIG_PATHS[1:]:
                if basefile.startswith(path):
                    basefile = basefile[len(path) + 1 :]

            file_hash = res.stdout.strip()
            hashes.setdefault(basefile, []).append(file_hash)

    with open(outpath, "w") as f:
        json.dump(hashes, f)


if __name__ == "__main__":
    gen_hashes(GENHASH_PATH)
