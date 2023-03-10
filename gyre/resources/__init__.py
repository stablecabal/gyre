
from gyres.resources.civitai import Civitai
import huggingface_hub

SOURCES = {
    "local"
    "civitai",
    "huggingface",
    "gdrive",
    "http"
}

TYPES = {
    "lora",
    "embedding",
    "image"
}

class ResourceProvider:

    def __init__(self, cache):
        self.cache = cache
    
    