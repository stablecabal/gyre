import inspect

import engines_pb2
import engines_pb2_grpc
import generation_pb2

from gyre.manager import EngineManager, ModelSet
from gyre.pipeline.samplers import sampler_properties
from gyre.services.exception_to_grpc import exception_to_grpc


class EnginesServiceServicer(engines_pb2_grpc.EnginesServiceServicer):
    _manager: EngineManager

    def __init__(self, manager):
        self._manager = manager

    @exception_to_grpc
    def ListEngines(self, request, context):
        engines = engines_pb2.Engines()

        # Add the no-op engine, which is always available
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
        engines.engine.append(info)

        status = self._manager.getStatus()
        for engine in self._manager.engines:
            if not (engine.is_engine and engine.visible and engine.task == "generate"):
                continue

            info = engines_pb2.EngineInfo()
            info.id = engine.id
            info.name = engine.name
            info.description = engine.description
            info.owner = "gyre"
            info.ready = status.get(engine.id, False)
            info.type = engines_pb2.EngineType.PICTURE

            class_obj = self._manager._import_class(engine.class_name)

            # Calculate samplers

            for sampler in sampler_properties(
                include_diffusers=getattr(class_obj, "_diffusers_capable", True),
                include_kdiffusion=getattr(class_obj, "_kdiffusion_capable", False),
            ):
                info.supported_samplers.append(engines_pb2.EngineSampler(**sampler))

            # Calculate supported inputs

            init_args = inspect.signature(class_obj.__init__).parameters.keys()
            call_args = inspect.signature(class_obj.__call__).parameters.keys()

            if "prompt" in call_args:
                info.accepted_prompt_artifacts.append(generation_pb2.ARTIFACT_TEXT)
            if "init_image" in call_args:
                info.accepted_prompt_artifacts.append(generation_pb2.ARTIFACT_IMAGE)
            if "mask_image" in call_args:
                info.accepted_prompt_artifacts.append(generation_pb2.ARTIFACT_MASK)
            if "depth_map" in call_args:
                info.accepted_prompt_artifacts.append(generation_pb2.ARTIFACT_DEPTH)
            if "lora" in call_args:
                info.accepted_prompt_artifacts.append(generation_pb2.ARTIFACT_LORA)

            # Add to list of engines

            engines.engine.append(info)

        return engines
