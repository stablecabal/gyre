from gyre.pipeline.model_utils import GPUExclusionSet, clone_model

NO_DEFAULT_SUPPLIED = object()


class HintsetManager:
    def __init__(self, hints=[], device="cpu", aligner=None):
        self.hints: list[dict] = hints
        self.device = device
        self.aligner = aligner

    def add_hint_handler(self, model, types, priority=100):
        self.hints.append({"model": model, "types": types, "priority": priority})

    def with_device(self, device):
        return HintsetManager(self.hints, device=device)

    def with_aligner(self, aligner):
        return HintsetManager(self.hints, aligner=aligner)

    def __align(self, model):
        if self.aligner is not None:
            return self.aligner(model)
        else:
            return clone_model(model, self.device)

    def for_type(self, type, default=NO_DEFAULT_SUPPLIED):
        for hint in sorted(self.hints, key=lambda hint: hint["priority"], reverse=True):
            if type in hint["types"]:
                return self.__align(hint["model"])

        if default is NO_DEFAULT_SUPPLIED:
            raise KeyError(f"No hint handler for type {type}")
        else:
            return default
