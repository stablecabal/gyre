from enum import Enum, unique


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


@unique
class ExtraCondition(Enum):
    sketch = 0
    keypose = 1
    seg = 2
    depth = 3
    canny = 4
    style = 5
    color = 6
    openpose = 7
