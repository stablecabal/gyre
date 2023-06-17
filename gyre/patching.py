import functools
import inspect


def patch_module_references(item, **patch):
    container_module = inspect.getmodule(item)

    # Handle the case of partial or other wrapped callables
    # (only for functools - other wrapper will break this function)
    if container_module is functools:
        container_module = inspect.getmodule(item.func)

    for k, v in patch.items():
        setattr(container_module, k, v)


def patch_setrlimit():
    import resource

    original_setrlimit = resource.setrlimit

    def wrapped_setrlimit(*args, **kwargs):
        try:
            return original_setrlimit(*args, **kwargs)
        except Exception as e:
            print("setrlimit failed")
            print("Arguments: ", args, kwargs)
            print("Exception: ", e)

    resource.setrlimit = wrapped_setrlimit
