import logging

from gyre.pipeline.model_utils import clone_model

logger = logging.getLogger(__name__)

NO_DEFAULT_SUPPLIED = object()


class HintsetManager:
    def __init__(self, hints=None, device="cpu", aligner=None):
        self.hints: list[dict] = [] if hints is None else hints
        self.device = device
        self.aligner = aligner

    def add_hint_handler(self, name, models, types, priority=100):
        self.hints.append(
            {"name": name, "models": models, "types": types, "priority": priority}
        )

    def with_device(self, device):
        return HintsetManager(self.hints, device=device)

    def with_aligner(self, aligner):
        return HintsetManager(self.hints, aligner=aligner)

    def __default_aligner(self, model):
        return clone_model(model, self.device)

    def __align(self, models):
        aligner = self.aligner or self.__default_aligner
        return {name: aligner(model) for name, model in models.items()}

    def for_type(self, type, default=NO_DEFAULT_SUPPLIED):
        for hint in sorted(self.hints, key=lambda hint: hint["priority"], reverse=True):
            if type in hint["types"]:
                logger.debug(f"Selected {hint['name']} for {type}")
                return self.__align(hint["models"])

        if default is NO_DEFAULT_SUPPLIED:
            raise KeyError(f"No hint handler for type {type}")
        else:
            return default
