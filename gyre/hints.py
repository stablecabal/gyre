from gyre.pipeline.model_utils import GPUExclusionSet, clone_model

NO_DEFAULT_SUPPLIED = object()


class HintsetManager:
    def __init__(self, hints=[], device="cpu"):
        self.hints: list[dict] = hints
        self.device = device

    def add_hint_handler(self, model, types, priority=100):
        self.hints.append({"model": model, "types": types, "priority": priority})

    def on(self, device):
        return HintsetManager(self.hints, device)

    def for_type(self, type, default=NO_DEFAULT_SUPPLIED):
        for hint in sorted(self.hints, key=lambda hint: hint["priority"], reverse=True):
            if type in hint["types"]:
                return clone_model(hint["model"], self.device)

        if default is NO_DEFAULT_SUPPLIED:
            raise KeyError(f"No hint handler for type {type}")
        else:
            return default
