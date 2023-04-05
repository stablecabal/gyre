import json
import uuid
from base64 import b64encode
from itertools import chain

import grpc
import multipart
import regex
from engines_pb2 import GENERATE, UPSCALE, Engines, EngineType, ListEnginesRequest
from generation_pb2 import (
    ARTIFACT_IMAGE,
    ARTIFACT_MASK,
    CHANNEL_A,
    CHANNEL_DISCARD,
    Artifact,
    DiffusionSampler,
    GuidanceInstanceParameters,
    ImageAdjustment,
    ImageAdjustment_Channels,
    ImageAdjustment_Invert,
    Prompt,
    PromptParameters,
    Request,
    StepParameter,
)
from twisted.web import resource
from twisted.web.resource import NoResource

from gyre.http.grpc_gateway_controller import GRPCServiceBridgeController
from gyre.http.json_api_controller import JSONError, UnsupportedMediaType


def number(data, attr, type, default=None, minVal=None, maxVal=None):
    val = type(data.get(attr, default))
    if minVal is not None and val < minVal:
        raise ValueError(f"{attr} may not be less than {minVal}, but {val} passed")
    if maxVal is not None and val > maxVal:
        raise ValueError(f"{attr} may not be more than {maxVal}, but {val} passed")

    return val


class StabilityRESTAPI_EnginesController(GRPCServiceBridgeController):
    def handle_GET(self, request, _):
        generators: Engines = self.servicer.ListEngines(
            ListEnginesRequest(task_group=GENERATE), request.grpc_context
        )
        upscalers: Engines = self.servicer.ListEngines(
            ListEnginesRequest(task_group=UPSCALE), request.grpc_context
        )

        res = []

        for engine in chain(generators.engine, upscalers.engine):
            res.append(
                dict(
                    id=engine.id,
                    name=engine.name,
                    description=engine.description,
                    type=EngineType.Name(engine.type),
                )
            )

        return {"engines": res}


class StabilityRESTAPI_ImageController(GRPCServiceBridgeController):
    preferred_return_type = "image/png"
    return_types = {"application/json", "image/png"}

    def __init__(self, servicer, engineid, gentype):
        super().__init__()

        self._engineid = engineid
        self._gentype = gentype
        self.add_servicer(servicer)

    text_prompt_rx = r"text_prompts\[([0-9]+)\]\[(text|weight)\]"

    def decode_GET(self, request):
        text_prompts = {}
        data = {}

        # TODO: This is duplicated from render_POST

        for k, v in request.args.items():
            name = k.decode("utf-8")
            value = v[0].decode("utf-8")

            if match := regex.match(self.text_prompt_rx, name):
                idx, label = int(match[1]), match[2]
                prompt = text_prompts.setdefault(idx, {})
                prompt[label] = value
            else:
                data[name] = value

        if text_prompts:
            keys = list(text_prompts.keys())
            keys.sort()
            data["text_prompts"] = [text_prompts[k] for k in keys]

        return data

    def decode_POST(self, request):
        content_type_header = request.getHeader("content-type")

        if content_type_header:
            content_type, options = multipart.parse_options_header(content_type_header)

            if content_type == "application/json":
                data = json.load(request.content)
                # Handle v1alpha-style "nested options" structure
                if "options" in data:
                    data.update(data.pop("options"))
                return data

            if content_type == "multipart/form-data":
                parser = multipart.MultipartParser(request.content, options["boundary"])
                data = {}
                text_prompts = {}
                for part in parser:
                    if part.name in {"image", "init_image", "mask_image"}:
                        if part.content_type != "image/png":
                            raise JSONError(500, "Images must be of type image/png")
                        data[part.name] = part.raw
                    elif part.name == "options":
                        data.update(json.load(part.file))
                    elif match := regex.match(self.text_prompt_rx, part.name):
                        idx, label = int(match[1]), match[2]
                        prompt = text_prompts.setdefault(idx, {})
                        prompt[label] = part.value
                    else:
                        data[part.name] = part.value

                if text_prompts:
                    keys = list(text_prompts.keys())
                    keys.sort()
                    data["text_prompts"] = [text_prompts[k] for k in keys]

                return data

        raise UnsupportedMediaType()

    def encode_image_png(self, request, answers):
        if request.grpc_context.code != grpc.StatusCode.OK:
            raise grpc.RpcError()

        for answer in answers:
            for artifact in answer.artifacts:
                if artifact.mime == "image/png":
                    request.setHeader("finish-reason", str(artifact.finish_reason))
                    request.setHeader("seed", str(artifact.seed))
                    return artifact.binary

        raise JSONError(
            500,
            "No appropriate image artifact found in response from generation service",
        )

    def encode_application_json(self, request, answers):
        if request.grpc_context.code != grpc.StatusCode.OK:
            raise grpc.RpcError()

        data = []

        for answer in answers:
            for artifact in answer.artifacts:
                if artifact.mime == "image/png":
                    data.append(
                        {
                            "base64": b64encode(artifact.binary).decode("ascii"),
                            "finishReason": artifact.finish_reason,
                            "seed": artifact.seed,
                        }
                    )

        if data:
            return json.dumps(data)

        raise JSONError(
            500,
            "No appropriate image artifact found in response from generation service",
        )

    def _image_to_prompt(
        self, image, init: bool = False, mask: bool = False, adjustments=[]
    ) -> Prompt:
        if init and mask:
            raise ValueError("init and mask cannot both be True")

        artifact = Artifact(
            type=ARTIFACT_MASK if mask else ARTIFACT_IMAGE, binary=image
        )

        for adjustment in adjustments:
            artifact.adjustments.append(adjustment)

        return Prompt(
            artifact=artifact,
            parameters=PromptParameters(init=init),
        )

    def _ia_alpha2rgb(self):
        return ImageAdjustment(
            channels=ImageAdjustment_Channels(
                r=CHANNEL_A,
                g=CHANNEL_A,
                b=CHANNEL_A,
                a=CHANNEL_DISCARD,
            )
        )

    def _ia_invert(self):
        return ImageAdjustment(invert=ImageAdjustment_Invert())

    def _mask_to_prompt(self, init_image, mask_image, mask_source):
        # With init_image_alpha, pull alpha from init_image into r,g,b then invert
        if mask_source == "INIT_IMAGE_ALPHA":
            return self._image_to_prompt(
                init_image,
                mask=True,
                adjustments=[self._ia_alpha2rgb(), self._ia_invert()],
            )

        else:
            if not mask_image:
                raise ValueError(
                    f"mask_source is {mask_source} but no mask_image was provided"
                )

            # With mask_image_white we can just use mask as-is
            if mask_source == "MASK_IMAGE_WHITE":
                return self._image_to_prompt(mask_image, mask=True)

            # With mask_image_black, invert the mask
            elif mask_source == "MASK_IMAGE_BLACK":
                return self._image_to_prompt(
                    mask_image, mask=True, adjustments=[self._ia_invert()]
                )

            else:
                raise ValueError(f"Unknown mask_source {mask_source}")

    def _prepare_request(self, data, is_png=False):
        raise NotImplementedError()

    def handle_BOTH(self, http_request, data):
        accept_header = http_request.getHeader("accept")
        is_png = accept_header == "image/png"

        generate_request = self._prepare_request(data, is_png)

        def generate(next):
            try:
                next(
                    self.servicer.Generate(generate_request, http_request.grpc_context)
                )
            except Exception as e:
                next(None, e)

        return generate


class StabilityRESTAPI_GenerationController(StabilityRESTAPI_ImageController):
    def _prepare_request(self, data, is_png=False):
        request = Request(
            engine_id=self._engineid.decode("utf-8"), request_id=str(uuid.uuid4())
        )
        parameters = StepParameter()

        # -- init_image

        init_image = data.pop("init_image", None)

        if self._gentype == b"text-to-image":
            if init_image:
                raise ValueError("Don't pass init_image to text-to-image")
        else:
            if not init_image:
                raise ValueError(f"{self._gentype} requires init_image")

            request.prompt.append(self._image_to_prompt(init_image, init=True))

        # -- mask_source

        mask_image = data.pop("mask_image", None)
        mask_source = data.get("mask_source", "").upper()

        if self._gentype != b"masking":
            if mask_source:
                raise ValueError(f"Don't pass mask_source to {self._gentype}")
        else:
            request.prompt.append(
                self._mask_to_prompt(init_image, mask_image, mask_source)
            )

        # -- cfg_scale

        parameters.sampler.cfg_scale = number(data, "cfg_scale", float, 7, 0, 35)

        # -- clip_guidance_preset

        if data.get("clip_guidance_preset", "NONE").upper() != "NONE":
            guidance_parameters = GuidanceInstanceParameters()
            guidance_parameters.guidance_strength = 0.333
            parameters.guidance.instances.append(guidance_parameters)

        # -- height

        if not init_image:
            request.image.height = number(data, "height", int, 512, 512, 2048)

        # -- sampler

        sampler_str = "SAMPLER_K_DPMPP_SDE"
        if "sampler" in data:
            sampler_str = "SAMPLER_" + str(data["sampler"]).upper()

        request.image.transform.diffusion = DiffusionSampler.Value(sampler_str)

        # -- samples

        if is_png:
            request.image.samples = number(data, "samples", int, 1, 1, 1)
        else:
            request.image.samples = number(data, "samples", int, 1, 1, 10)

        # -- seed

        if "seed" in data:
            request.image.seed.append(number(data, "seed", int, 0, 0, 2147483647))

        # -- init_image_mode (automatically determine for all modes where possible)

        has_image_strength = "image_strength" in data
        has_schedule = "step_schedule_start" in data or "step_schedule_end" in data

        if has_image_strength and not has_schedule:
            init_image_mode = "IMAGE_STRENGTH"
        elif has_schedule and not has_image_strength:
            init_image_mode = "STEP_SCHEDULE"
        else:
            default_iim = "IMAGE_STRENGTH" if init_image else "STEP_SCHEDULE"
            init_image_mode = data.get("init_image_mode", default_iim).upper()

        if init_image_mode == "IMAGE_STRENGTH":

            # -- image_strength

            parameters.schedule.end = 0
            parameters.schedule.start = 1 - number(
                data, "image_strength", float, 0.35, 0, 1
            )

        else:

            # -- step_schedule_end

            parameters.schedule.end = number(data, "step_schedule_end", float, 0, 0, 1)

            # -- step_schedule_start

            parameters.schedule.start = number(
                data, "step_schedule_start", float, 1, 0, 1
            )

        # -- steps

        request.image.steps = number(data, "steps", int, 50, 10, 150)

        # -- text_prompts

        for prompt in data["text_prompts"]:
            rp = Prompt()
            rp.text = str(prompt["text"])
            rp.parameters.weight = float(prompt.get("weight", 1.0))
            request.prompt.append(rp)

        # -- width

        if not init_image:
            request.image.width = number(data, "width", int, 512, 512, 2048)

        request.image.parameters.append(parameters)
        return request


class StabilityRESTAPI_UpscaleController(StabilityRESTAPI_ImageController):
    def _prepare_request(self, data, is_png=False):
        request = Request(
            engine_id=self._engineid.decode("utf-8"), request_id=str(uuid.uuid4())
        )

        # -- image

        image = data.pop("image", None)

        if image is None:
            raise ValueError(f"{self._gentype} requires init_image")

        request.prompt.append(self._image_to_prompt(image, init=True))

        # -- width

        if "width" in data:
            request.image.width = number(data, "width", int, 512, 512, 2048)

        # -- height

        if "height" in data:
            request.image.height = number(data, "height", int, 512, 512, 2048)

        return request


class StabilityRESTAPI_GenerationRouter(GRPCServiceBridgeController):
    def __init__(self):
        self._servicer = None
        super().__init__()

    def add_servicer(self, servicer):
        self._servicer = servicer

    def getChild(self, path, request):
        if not self._servicer:
            return JSONError(503, "Not ready yet")

        engineid = path
        gentype = None

        engine_spec = self._servicer._manager._find_spec(id=engineid.decode("utf-8"))

        if engine_spec is None:
            return JSONError(404, "No such engine")

        if request.postpath:
            gentype = request.postpath.pop(0)

        if gentype == b"image-to-image" and request.postpath:
            gentype = request.postpath.pop(0)

        if gentype in {b"text-to-image", b"image-to-image", b"masking"}:
            if engine_spec.task != "generate":
                return JSONError(400, "Engine is not a generate task engine")

            return StabilityRESTAPI_GenerationController(
                self._servicer,
                engineid,
                gentype,
            )

        if gentype in {b"upscale"}:
            if engine_spec.task != "upscaler":
                return JSONError(400, "Engine is not a upscaler task engine")

            return StabilityRESTAPI_UpscaleController(
                self._servicer,
                engineid,
                gentype,
            )

        return JSONError(404, "Unknown engine task")


class StabilityRESTAPIRouter(resource.Resource):
    def __init__(self):
        super().__init__()

        self.engines_router = StabilityRESTAPI_EnginesController()
        self.generation_router = StabilityRESTAPI_GenerationRouter()

    def getChild(self, path, request):
        if path == b"engines":
            return self.engines_router
        if path == b"generation":
            return self.generation_router

        return NoResource()

    def render(self, request):
        return NoResource().render(request)

    def add_EnginesServiceServicer(self, engines_servicer):
        self.engines_router.add_servicer(engines_servicer)

    def add_GenerationServiceServicer(self, generation_servicer):
        self.generation_router.add_servicer(generation_servicer)
