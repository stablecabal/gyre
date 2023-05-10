import inspect

import engines_pb2
import engines_pb2_grpc
import generation_pb2

from gyre.manager import EngineManager, ModelSet
from gyre.pipeline import pipeline_meta
from gyre.pipeline.samplers import sampler_properties
from gyre.services.exception_to_grpc import exception_to_grpc

TASK_GROUPS = {
    engines_pb2.GENERATE: {"generate"},
    engines_pb2.UPSCALE: {"upscaler"},
    engines_pb2.UTILITY: {"decode_latents", "noop"},
    engines_pb2.HINTER: {
        "depth",
        "edge_detection",
        "segmentation",
        "pose",
        "background-removal",
    },
}


class EnginesServiceServicer(engines_pb2_grpc.EnginesServiceServicer):
    _manager: EngineManager

    def __init__(self, manager):
        self._manager = manager

    def engines_of_task_group(self, task_group):
        tasks = TASK_GROUPS[task_group]

        for engine in self._manager.engines:
            if engine.is_engine and engine.visible and engine.task in tasks:
                yield engine

    def build_noop_info(self):
        info = engines_pb2.EngineInfo()
        info.id = "noop"
        info.name = "No-op engine"
        info.description = (
            "Does nothing, just returns the init image without further processing."
        )
        info.owner = "gyre"
        info.ready = True
        info.type = engines_pb2.EngineType.PICTURE
        info.accepted_prompt_artifacts.append(generation_pb2.ARTIFACT_IMAGE)
        info.task = "noop"

        return info

    def build_engine_info(self, engine):
        info = engines_pb2.EngineInfo()
        info.id = engine.id
        info.name = engine.name or "Unnamed"
        info.description = engine.description or "No description"
        info.owner = "gyre"
        info.ready = self._manager.getStatusByID(engine.id)
        info.type = engines_pb2.EngineType.PICTURE
        info.task = engine.task

        class_obj = self._manager._import_class(engine.class_name)

        # Get some introspected info

        init_args = inspect.signature(class_obj.__init__).parameters.keys()
        call_args = inspect.signature(class_obj.__call__).parameters.keys()

        # Calculate samplers

        if "scheduler" in init_args:
            meta = pipeline_meta.get_meta(class_obj)

            for sampler in sampler_properties(
                include_diffusers=meta.get("diffusers_capable", True),
                include_kdiffusion=meta.get("kdiffusion_capable", False),
            ):
                info.supported_samplers.append(engines_pb2.EngineSampler(**sampler))

        # Calculate supported inputs

        if "prompt" in call_args:
            info.accepted_prompt_artifacts.append(generation_pb2.ARTIFACT_TEXT)
        if "image" in call_args:
            info.accepted_prompt_artifacts.append(generation_pb2.ARTIFACT_IMAGE)
        if "mask_image" in call_args:
            info.accepted_prompt_artifacts.append(generation_pb2.ARTIFACT_MASK)
        if "depth_map" in call_args:
            info.accepted_prompt_artifacts.append(generation_pb2.ARTIFACT_DEPTH)
        if "lora" in call_args:
            info.accepted_prompt_artifacts.append(generation_pb2.ARTIFACT_LORA)
        if "token_embeddings" in call_args:
            info.accepted_prompt_artifacts.append(
                generation_pb2.ARTIFACT_TOKEN_EMBEDDING
            )
        if "hint_images" in call_args:
            info.accepted_prompt_artifacts.append(generation_pb2.ARTIFACT_HINT_IMAGE)

        # Calculate hints

        if engine.hintset:
            supported_hint_types = {}

            for name, hintset in self._manager._build_hintset(
                engine.hintset, with_models=False
            ).items():
                for type in hintset["types"]:
                    supported_hint_types.setdefault(type, set()).add(name)

            for type, providers in supported_hint_types.items():
                info.accepted_hint_types.append(
                    engines_pb2.EngineHintImageType(type=type, provider=list(providers))
                )

        return info

    @exception_to_grpc
    def ListEngines(self, request, context):
        engines = engines_pb2.Engines()

        if request.task_group == engines_pb2.UTILITY:
            engines.engine.append(self.build_noop_info())

        for engine in self.engines_of_task_group(request.task_group):
            # Add to list of engines
            engines.engine.append(self.build_engine_info(engine))

        return engines
