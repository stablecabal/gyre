import glob
import hashlib
import json
import os
import re
import shutil
import subprocess

import yaml

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIST_CONFIG_PATH = os.path.join(BASE_PATH, "gyre/config")
GENHASH_PATH = os.path.join(DIST_CONFIG_PATH, "dist_hashes.json")

# All the paths config has been distributed in historically
DIST_CONFIG_PATHS = ["engines.yaml", "sdgrpcserver/config", "gyre/config"]


ID_KEYS = {"id", "model_id", "hintset_id"}


class Template:
    def __init__(self, id, extends, abstract, params, mappings: list[dict]):
        self.id = id
        self.extends = extends
        self.abstract = abstract
        self.params = params
        self.mappings = mappings


class Params:
    def __init__(self, mapping: dict):
        self.mapping = mapping


def LoaderFactory(sources: list[str], context: dict, cache: dict):
    if "loader" in cache:
        return cache["loader"]

    class Loader(yaml.SafeLoader):
        def __init__(self, stream):
            self._root = None
            if stream_name := getattr(stream, "name", None):
                self._root = os.path.split(stream_name)[0]

            super(Loader, self).__init__(stream)

        def _parse_blockname(self, tag):
            tag = tag.strip()

            if tag[-1] == ")":
                return tag[:-1].split("(", 1)
            else:
                return tag, None

        def template(self, tag, node):
            ext, name = node.tag.lstrip("!@").split("/")
            if isinstance(node, yaml.MappingNode):
                params = Params(self.construct_mapping(node, True))
                mappings = []
            else:
                val = self.construct_sequence(node, True)
                try:
                    params = next((obj for obj in val if isinstance(obj, Params)))
                except StopIteration:
                    params = Params({})
                mappings = [obj for obj in val if not isinstance(obj, Params)]

            abstract = node.tag.startswith("!@")

            self.add_multi_constructor(f"!{name}/", Loader.template)
            self.add_multi_constructor(f"!@{name}/", Loader.template)

            return Template(name, ext, abstract, params, mappings)

        def params(self, node):
            val = self.construct_mapping(node, True)
            return Params(val)

        def include(self, tag, node):
            if self._root is None:
                raise ValueError("Can't !include from a yaml string, only from files")

            if tag:
                print(tag, context.get(tag.strip("()")))
                if not context.get(tag.strip("()")):
                    return

            # We insert any new files at the begining, do to a depth-first import
            paths = list(
                sorted(glob.glob(os.path.join(self._root, self.construct_scalar(node))))
            )

            sources.extend(paths)

        def merge(self, tag, node):
            return self.include(tag, node)

        def none(self, node):
            return None

    Loader.add_multi_constructor("!template/", Loader.template)
    Loader.add_multi_constructor("!@template/", Loader.template)
    Loader.add_constructor("!params", Loader.params)
    Loader.add_multi_constructor("!include", Loader.include)
    Loader.add_multi_constructor("!merge", Loader.merge)
    Loader.add_constructor("!none", Loader.none)

    cache["loader"] = Loader
    return Loader


def load_raw_yaml(paths, context):

    data = []
    sources = [*paths]
    cache = {}

    while sources:
        # Get the next source
        source = sources.pop(0)

        if source.endswith(".yaml") or source.endswith(".yml"):
            source = open(source, "rb")

        includes = []

        # Load it into docs & blocks (and extend source if there are includes)
        docs = list(
            yaml.load_all(
                source, Loader=LoaderFactory(includes, context=context, cache=cache)
            )
        )

        # Append all includes to the beginning so we do depth-first
        sources[0:0] = includes

        for doc in docs:
            if isinstance(doc, list):
                data.extend([item for item in doc if item is not None])
            else:
                data.append(doc)

    return data


class Bubble:
    def __init__(self, val):
        self.val = val


def walk_template(el, context):

    re_partial = r"{{>\s*(.*?)\s*}}"
    re_bool = r"{{([#^])\s*(.*?)\s*}}"
    re_fullvar = r"{{(.*?)}}$"
    re_var = r"{{(.*?)}}"

    def get_from_context(key: str):
        default = None

        if ":" in key:
            key, default = key.split(":", maxsplit=1)
            default = yaml.load(default, Loader=yaml.SafeLoader)

        return context.get(key, default)

    # Dictionary section
    if isinstance(el, dict):
        res = {}
        is_single = len(el) == 1

        for k, v in el.items():

            # Handle including partial
            if match := re.match(re_partial, k):
                ko = walk_template(k, context)
                if isinstance(ko, dict):
                    res.update(ko)
                elif is_single and isinstance(ko, list):
                    return Bubble(ko)
                elif ko:
                    raise ValueError(
                        f"Partial {match[1]} returned a non-false object of type {type(ko)}, "
                        "which we don't know how to add to a dict"
                    )

            # Handle boolean check
            elif match := re.match(re_bool, k):
                ko = walk_template(k, context)
                if ko:
                    vo = walk_template(v, context)
                    if isinstance(vo, dict):
                        res.update(vo)
                    elif is_single and isinstance(vo, list):
                        return Bubble(vo)
                    elif vo:
                        raise ValueError(
                            f"Bool check {match[2]} returned a non-false object of type {type(vo)}, "
                            "which we don't know how to add to a dict"
                        )

            # Handle regular replacement
            else:
                ko = walk_template(k, context)
                vo = walk_template(v, context)

                if isinstance(vo, Bubble):
                    raise ValueError(
                        f"Got a bubbled list from {k}, which we don't know how to handle"
                    )

                res[ko] = vo

        return res

    # List section
    elif isinstance(el, list):
        res = []
        for i, v in enumerate(el):
            vo = walk_template(v, context)
            if isinstance(vo, Bubble):
                res.extend(vo.val)
            elif vo:
                res.append(vo)

        return res

    # String section
    elif isinstance(el, str):
        if match := re.match(re_partial, el):
            return get_from_context(match[1])
        elif match := re.match(re_bool, el):
            val = get_from_context(match[2])
            return not val if match[1] == "^" else val
        elif match := re.match(re_fullvar, el):
            return get_from_context(match[1])
        else:
            return re.sub(re_var, lambda m: get_from_context(m[1]), el)

    else:
        return el


def merge_dict(mapping, *updating_mappings):
    res = mapping.copy()

    for updating_mapping in updating_mappings:
        for k, v in updating_mapping.items():
            if k in res and isinstance(res[k], dict) and isinstance(v, dict):
                res[k] = merge_dict(res[k], v)
            elif v is None:
                del res[k]
            else:
                res[k] = v

    return res


def merge_dicts(items):
    res = []

    for item in items:
        if isinstance(item, dict) and (keys := ID_KEYS & set(item.keys())):
            key = keys.pop()
            for i, other in enumerate(res):
                if isinstance(other, dict) and other.get(key) == item[key]:
                    res[i] = merge_dict(other, item)
                    break
            else:
                res.append(item)
        elif item is not None:
            res.append(item)

    return res


def flatten_templates(items):
    res = []
    templates = {}

    for item in items:
        if isinstance(item, Template) and item.id in templates:
            templates[item.id].params.mapping = merge_dict(
                templates[item.id].params.mapping, item.params.mapping
            )
            templates[item.id].mappings = merge_dicts(
                templates[item.id].mappings + item.mappings
            )

        else:
            res.append(item)
            if isinstance(item, Template):
                templates[item.id] = item

    return res


def apply_templates(items):
    res = []
    templates = {}

    for item in items:
        if isinstance(item, Template):
            if item.abstract:
                templates[item.id] = item

            else:
                context = {"id": item.id}
                mappings = []

                template = item
                while template:
                    # Apply the context to the main template body
                    output = walk_template(template.mappings, context)

                    # And update the merged result
                    mappings = merge_dicts(output + mappings)

                    # Apply the existing context to any params the result
                    outparams = walk_template(template.params.mapping, context)
                    # and then update the context with
                    context = merge_dict(context, outparams)

                    if template.extends and template.extends != "template":
                        template = templates[template.extends]
                    else:
                        template = None

                res.extend(mappings)
        else:
            res.append(item)

    return res


def load(paths, context):
    data = load_raw_yaml(paths, context)
    data = flatten_templates(data)
    data = apply_templates(data)
    data = merge_dicts(data)

    # yaml.dump(data, stream=open("/weights/config.log", "w"))
    # sys.exit(-1)

    return data


def git_object_hash(bs: bytes):
    hasher = hashlib.sha1()
    hasher.update(b"blob ")
    hasher.update(bytes(str(len(bs)), "utf-8"))
    hasher.update(b"\0")
    hasher.update(bs)
    return hasher.hexdigest()


def check_and_update(config_path):

    with open(GENHASH_PATH, "r") as f:
        dist_hashes = json.load(f)
        for path, hashes in dist_hashes.items():

            exists = False
            update = True

            # Check the current config file to see if it matches a distribution hash
            # (if not we ingore it, as it's user edited)

            current_hash = None
            current_path = os.path.join(config_path, path)

            if os.path.isfile(current_path):
                exists = True
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
            elif update and exists:
                print(
                    "Config file",
                    path,
                    "appears to be obsolete and can be removed."
                    "(For safety reasons, Gyre won't do this for you, you'll need to do it manually).",
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
