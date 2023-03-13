import importlib
import importlib.util
import os
import sys
import types

SRC_MODULES = ["midas", "mmsegmentation"]

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
            m = types.ModuleType(full_name, f"Fake {module_name} stub")
            print(full_name)
            sys.modules[full_name] = m
            parent = full_name


def kdiffusion_direct_import(name):
    if "k_diffusion" not in sys.modules:
        m = types.ModuleType("k_diffusion")
        sys.modules["k_diffusion"] = m

    module_name = f"k_diffusion.{name}"
    file_path = os.path.join(src_dir, "k-diffusion/k_diffusion", f"{name}.py")

    # From https://docs.python.org/3/library/importlib.html#importing-a-source-file-directly
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    # setattr(sys.modules["k_diffusion"], name, module)
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

    # Stub out mmsegmentation libraries we don't use
    stub_dependancy("matplotlib.pyplot")

    for name in SRC_MODULES:
        sys.path.append(os.path.join(src_dir, name))

    __injected = True
