"""
@generated by mypy-protobuf.  Do not edit manually!
isort:skip_file
"""
import builtins
import collections.abc
import generation_pb2
import google.protobuf.descriptor
import google.protobuf.internal.containers
import google.protobuf.internal.enum_type_wrapper
import google.protobuf.message
import sys
import typing

if sys.version_info >= (3, 10):
    import typing as typing_extensions
else:
    import typing_extensions

DESCRIPTOR: google.protobuf.descriptor.FileDescriptor

class _EngineType:
    ValueType = typing.NewType("ValueType", builtins.int)
    V: typing_extensions.TypeAlias = ValueType

class _EngineTypeEnumTypeWrapper(google.protobuf.internal.enum_type_wrapper._EnumTypeWrapper[_EngineType.ValueType], builtins.type):
    DESCRIPTOR: google.protobuf.descriptor.EnumDescriptor
    TEXT: _EngineType.ValueType  # 0
    PICTURE: _EngineType.ValueType  # 1
    AUDIO: _EngineType.ValueType  # 2
    VIDEO: _EngineType.ValueType  # 3
    CLASSIFICATION: _EngineType.ValueType  # 4
    STORAGE: _EngineType.ValueType  # 5

class EngineType(_EngineType, metaclass=_EngineTypeEnumTypeWrapper):
    """Possible engine type"""

TEXT: EngineType.ValueType  # 0
PICTURE: EngineType.ValueType  # 1
AUDIO: EngineType.ValueType  # 2
VIDEO: EngineType.ValueType  # 3
CLASSIFICATION: EngineType.ValueType  # 4
STORAGE: EngineType.ValueType  # 5
global___EngineType = EngineType

class _EngineTokenizer:
    ValueType = typing.NewType("ValueType", builtins.int)
    V: typing_extensions.TypeAlias = ValueType

class _EngineTokenizerEnumTypeWrapper(google.protobuf.internal.enum_type_wrapper._EnumTypeWrapper[_EngineTokenizer.ValueType], builtins.type):
    DESCRIPTOR: google.protobuf.descriptor.EnumDescriptor
    GPT2: _EngineTokenizer.ValueType  # 0
    PILE: _EngineTokenizer.ValueType  # 1

class EngineTokenizer(_EngineTokenizer, metaclass=_EngineTokenizerEnumTypeWrapper): ...

GPT2: EngineTokenizer.ValueType  # 0
PILE: EngineTokenizer.ValueType  # 1
global___EngineTokenizer = EngineTokenizer

class _EngineTaskGroup:
    ValueType = typing.NewType("ValueType", builtins.int)
    V: typing_extensions.TypeAlias = ValueType

class _EngineTaskGroupEnumTypeWrapper(google.protobuf.internal.enum_type_wrapper._EnumTypeWrapper[_EngineTaskGroup.ValueType], builtins.type):
    DESCRIPTOR: google.protobuf.descriptor.EnumDescriptor
    GENERATE: _EngineTaskGroup.ValueType  # 0
    UPSCALE: _EngineTaskGroup.ValueType  # 1
    UTILITY: _EngineTaskGroup.ValueType  # 2
    HINTER: _EngineTaskGroup.ValueType  # 3

class EngineTaskGroup(_EngineTaskGroup, metaclass=_EngineTaskGroupEnumTypeWrapper): ...

GENERATE: EngineTaskGroup.ValueType  # 0
UPSCALE: EngineTaskGroup.ValueType  # 1
UTILITY: EngineTaskGroup.ValueType  # 2
HINTER: EngineTaskGroup.ValueType  # 3
global___EngineTaskGroup = EngineTaskGroup

@typing_extensions.final
class EngineSampler(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    SAMPLER_FIELD_NUMBER: builtins.int
    SUPPORTS_ETA_FIELD_NUMBER: builtins.int
    SUPPORTS_CHURN_FIELD_NUMBER: builtins.int
    SUPPORTS_SIGMA_LIMITS_FIELD_NUMBER: builtins.int
    SUPPORTS_KARRAS_RHO_FIELD_NUMBER: builtins.int
    SUPPORTED_NOISE_TYPES_FIELD_NUMBER: builtins.int
    sampler: generation_pb2.DiffusionSampler.ValueType
    supports_eta: builtins.bool
    supports_churn: builtins.bool
    supports_sigma_limits: builtins.bool
    supports_karras_rho: builtins.bool
    @property
    def supported_noise_types(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[generation_pb2.SamplerNoiseType.ValueType]: ...
    def __init__(
        self,
        *,
        sampler: generation_pb2.DiffusionSampler.ValueType = ...,
        supports_eta: builtins.bool = ...,
        supports_churn: builtins.bool = ...,
        supports_sigma_limits: builtins.bool = ...,
        supports_karras_rho: builtins.bool = ...,
        supported_noise_types: collections.abc.Iterable[generation_pb2.SamplerNoiseType.ValueType] | None = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["sampler", b"sampler", "supported_noise_types", b"supported_noise_types", "supports_churn", b"supports_churn", "supports_eta", b"supports_eta", "supports_karras_rho", b"supports_karras_rho", "supports_sigma_limits", b"supports_sigma_limits"]) -> None: ...

global___EngineSampler = EngineSampler

@typing_extensions.final
class EngineHintImageType(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    TYPE_FIELD_NUMBER: builtins.int
    PROVIDER_FIELD_NUMBER: builtins.int
    type: builtins.str
    @property
    def provider(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.str]: ...
    def __init__(
        self,
        *,
        type: builtins.str = ...,
        provider: collections.abc.Iterable[builtins.str] | None = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["provider", b"provider", "type", b"type"]) -> None: ...

global___EngineHintImageType = EngineHintImageType

@typing_extensions.final
class EngineInfo(google.protobuf.message.Message):
    """Engine info struct"""

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    ID_FIELD_NUMBER: builtins.int
    OWNER_FIELD_NUMBER: builtins.int
    READY_FIELD_NUMBER: builtins.int
    TYPE_FIELD_NUMBER: builtins.int
    TOKENIZER_FIELD_NUMBER: builtins.int
    NAME_FIELD_NUMBER: builtins.int
    DESCRIPTION_FIELD_NUMBER: builtins.int
    SUPPORTED_SAMPLERS_FIELD_NUMBER: builtins.int
    ACCEPTED_PROMPT_ARTIFACTS_FIELD_NUMBER: builtins.int
    ACCEPTED_HINT_TYPES_FIELD_NUMBER: builtins.int
    TASK_FIELD_NUMBER: builtins.int
    id: builtins.str
    owner: builtins.str
    ready: builtins.bool
    type: global___EngineType.ValueType
    tokenizer: global___EngineTokenizer.ValueType
    name: builtins.str
    description: builtins.str
    @property
    def supported_samplers(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___EngineSampler]: ...
    @property
    def accepted_prompt_artifacts(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[generation_pb2.ArtifactType.ValueType]: ...
    @property
    def accepted_hint_types(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___EngineHintImageType]: ...
    task: builtins.str
    def __init__(
        self,
        *,
        id: builtins.str = ...,
        owner: builtins.str = ...,
        ready: builtins.bool = ...,
        type: global___EngineType.ValueType = ...,
        tokenizer: global___EngineTokenizer.ValueType = ...,
        name: builtins.str = ...,
        description: builtins.str = ...,
        supported_samplers: collections.abc.Iterable[global___EngineSampler] | None = ...,
        accepted_prompt_artifacts: collections.abc.Iterable[generation_pb2.ArtifactType.ValueType] | None = ...,
        accepted_hint_types: collections.abc.Iterable[global___EngineHintImageType] | None = ...,
        task: builtins.str = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["accepted_hint_types", b"accepted_hint_types", "accepted_prompt_artifacts", b"accepted_prompt_artifacts", "description", b"description", "id", b"id", "name", b"name", "owner", b"owner", "ready", b"ready", "supported_samplers", b"supported_samplers", "task", b"task", "tokenizer", b"tokenizer", "type", b"type"]) -> None: ...

global___EngineInfo = EngineInfo

@typing_extensions.final
class ListEnginesRequest(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    TASK_GROUP_FIELD_NUMBER: builtins.int
    task_group: global___EngineTaskGroup.ValueType
    """Empty"""
    def __init__(
        self,
        *,
        task_group: global___EngineTaskGroup.ValueType = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["task_group", b"task_group"]) -> None: ...

global___ListEnginesRequest = ListEnginesRequest

@typing_extensions.final
class Engines(google.protobuf.message.Message):
    """Engine info list"""

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    ENGINE_FIELD_NUMBER: builtins.int
    @property
    def engine(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___EngineInfo]: ...
    def __init__(
        self,
        *,
        engine: collections.abc.Iterable[global___EngineInfo] | None = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["engine", b"engine"]) -> None: ...

global___Engines = Engines
