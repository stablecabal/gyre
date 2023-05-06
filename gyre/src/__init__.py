import importlib
import importlib.util
import os
import sys
import types
import warnings

SRC_MODULES = [
    "xtcocoapi",
    "midas",
    "mmsegmentation",
    "mmdetection",
    "mmpose",
    "picklemagic",
    "BasicSR",
    "ZoeDepth",
]

src_dir = os.path.dirname(__file__)


def stub_dependancy(module_name):
    try:
        importlib.import_module(module_name)
    except ModuleNotFoundError as e:
        parts = module_name.split(".")
        parent = None

        while parts:
            short_name = parts.pop(0)
            full_name = parent + "." + short_name if parent else short_name
            if full_name not in sys.modules:
                m = types.ModuleType(full_name, f"Fake {module_name} stub")
                sys.modules[full_name] = m
            parent = full_name

    return sys.modules[module_name]


def stub_basicsr_version():
    module = stub_dependancy("basicsr.version")

    with open(os.path.join(src_dir, "BasicSR", "VERSION"), "r") as f:
        SHORT_VERSION = f.read().strip()
    VERSION_INFO = [x if x.isdigit() else f'"{x}"' for x in SHORT_VERSION.split(".")]

    module.__version__ = "{}"
    module.__gitsha__ = "unknown"
    module.version_info = tuple(VERSION_INFO)


def kdiffusion_direct_import(name):
    base_path = os.path.join(src_dir, "k-diffusion/k_diffusion")

    if "k_diffusion" not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            "k_diffusion", os.path.join(base_path, "__init__.py")
        )
        m = types.ModuleType("k_diffusion")
        m.__spec__ = spec
        sys.modules["k_diffusion"] = m

    module_name = f"k_diffusion.{name}"
    file_path = os.path.join(base_path, f"{name}.py")

    # From https://docs.python.org/3/library/importlib.html#importing-a-source-file-directly
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    # setattr(sys.modules["k_diffusion"], name, module)
    spec.loader.exec_module(module)


def fix_mmpose_version_constraint():
    """
    mmpose v0.29.0 has a constraint of <= 1.7.0 (and v1.0.0rc0 is >=v2.0.0).
    Rather than dealing with cross-version hell between various mmXXX packages,
    we just trick mmpose into thinking we have the right mmcv version
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import mmcv

    if mmcv.__version__ == "1.7.1":
        mmcv.__actual_version__ = mmcv.__version__
        mmcv.__version__ = "1.7.0"


class ClassStub:
    def __getattr__(self, k):
        return self

    def __call__(self, *args, **kwargs):
        pass


class CocotoolsRewriter:
    """
    This has two purposes:
    - Rewrite pycocotools to use xtcocotools
    - Replace xtcocotools.mask (and pycocotools.mask) with a stub (would otherwise require
    a compiled C module)
    """

    def find_module(self, fullname, path=None):
        if fullname in {"xtcocotools.mask", "pycocotools.mask"}:
            return ClassStub()

        if fullname.startswith("pycocotools"):
            return self

    def load_module(self, name):
        if name.startswith("pycocotools"):
            spec = importlib.util.find_spec("xt" + name[2:])
            sys.modules[name] = module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)


__injected = False


def inject_src_paths():
    global __injected

    if __injected:
        return

    # Load k_diffusion directly, as it's __init__ loads a bunch of files we don't
    # use and which pull in a bunch of dependancies. Note: order critical
    for name in ["utils", "sampling", "external"]:
        kdiffusion_direct_import(name)

    # Rewrite some problematic imports to do with pycocotools / xtcocotools
    sys.meta_path.insert(0, CocotoolsRewriter())

    # Stub out mmsegmentation libraries we don't use
    stub_dependancy("matplotlib.pyplot")
    stub_dependancy("matplotlib.collections.PatchCollection")
    stub_dependancy("matplotlib.patches.Polygon")

    # Add names to path
    for name in SRC_MODULES:
        sys.path.append(os.path.join(src_dir, name))

    # Fix BasicSR version being generated on package build
    stub_basicsr_version()

    # Hack around mmpose version constraint
    fix_mmpose_version_constraint()

    __injected = True
