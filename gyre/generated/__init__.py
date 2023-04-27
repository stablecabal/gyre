import os
import sys


def inject_generated_path():
    generated_path = os.path.dirname(__file__)
    sys.path.append(generated_path)
