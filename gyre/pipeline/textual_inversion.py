import logging

import torch

from gyre.pipeline.model_utils import (
    CloneToGPUHook,
    ModelHook,
    add_hook,
    get_hook,
    has_hook,
    is_hooked,
    remove_hook,
)

logger = logging.getLogger(__name__)


class ApplyEmbeddingHook(ModelHook):
    def __init__(self):
        self.additions: dict[int, torch.Tensor] = {}

    # If we're using a GPU hook, then we need to modify the embedding parameter _after_
    # it's been copied to the GPU - editing it beforehand would change the shared copy
    def pre_forward(self, module, *args, **kwargs):
        num_tokens, embedding_dim = module.num_embeddings, module.embedding_dim
        cur_num_tokens, cur_embedding_dim = module.weight.size()

        if num_tokens == cur_num_tokens:
            # Nothing to do, return
            return args, kwargs

        # Build new embeddings
        new_weight = torch.nn.parameter.Parameter(
            torch.empty(
                (num_tokens, embedding_dim),
                dtype=module.weight.dtype,
                device=module.weight.device,
            )
        )

        # initialize all new embeddings (in particular added tokens)
        # self._init_weights(new_embeddings)

        new_weight.data[:cur_num_tokens, :] = module.weight.data[:cur_num_tokens, :]
        for idx, tensor in self.additions.items():
            new_weight.data[idx] = tensor.to(module.weight.device)

        module._parameters["weight"] = new_weight

        return args, kwargs


def attach_multidim_handler(tokenizer):
    if not hasattr(tokenizer, "_multidim_mappings"):
        tokenizer._multidim_mappings = {}

    def replace_text(self, text):
        # If text is a list, recurse until it isn't
        if isinstance(text, list):
            return [replace_text(self, t) for t in text]

        # OK, so text is just a str now
        for token, replacement in self._multidim_mappings.items():
            text = text.replace(token, replacement)

        return text

    def patched_call(self, text, *args, **kwargs):
        text = replace_text(self, text)
        return self.__class__._call_one(self, text, *args, **kwargs)

    def patched_encode(self, text, *args, **kwargs):
        text = replace_text(self, text)
        return self.__class__.encode(self, text, *args, **kwargs)

    tokenizer._call_one = patched_call.__get__(tokenizer)
    tokenizer.encode = patched_encode.__get__(tokenizer)


def remove_multidim_handler(tokenizer):
    for attr in ("_multidim_mappings", "_call_one", "encode"):
        try:
            delattr(tokenizer, attr)
        except AttributeError:
            pass


def apply_ti_to_tokenizer(tokenizer, tensors: dict[str, torch.Tensor]):

    # Pre-process any multidim tokens

    flattened = {}

    for token, tensor in tensors.items():
        if len(tensor.shape) == 2:
            if not hasattr(tokenizer, "_multidim_mappings"):
                attach_multidim_handler(tokenizer)

            subbase = token.lstrip("<").rstrip(">")
            subtokens = [f"<{subbase}{i:02x}>" for i in range(0, tensor.shape[0])]

            for i, subtoken in enumerate(subtokens):
                flattened[subtoken] = tensor[i]

            tokenizer._multidim_mappings[token] = " ".join(subtokens)

        else:
            flattened[token] = tensor

    # Add the flattened tokens to the tokenizer

    # Note - must be a converted to a list or add_tokens will treat it like a single token
    if tokenizer.add_tokens(list(flattened.keys())) != len(flattened.keys()):
        logger.debug(f"Tokenizer token overlap when loading textual inversion")

    return flattened


def apply_ti_to_encoder(tokenizer, text_encoder, flattened: dict[str, torch.Tensor]):

    # Set up the text_encoder to accept the new embeddings

    orig_embedding = text_encoder.get_input_embeddings()
    new_embedding = text_encoder.resize_token_embeddings(len(tokenizer))

    if new_embedding is not orig_embedding and is_hooked(orig_embedding):
        add_hook(new_embedding, orig_embedding._hf_hook, replace=True)

    embedding_hook = None
    if has_hook(orig_embedding, CloneToGPUHook):
        embedding_hook = get_hook(new_embedding, ApplyEmbeddingHook)
        if embedding_hook is False:
            embedding_hook = ApplyEmbeddingHook()
            add_hook(new_embedding, embedding_hook)

    # Now add them to the embedding

    dtype = text_encoder.get_input_embeddings().weight.dtype

    for token, tensor in flattened.items():
        token_id = tokenizer.convert_tokens_to_ids(token)

        if embedding_hook:
            embedding_hook.additions[token_id] = tensor.to(dtype)

        if new_embedding.weight.device != torch.device("meta"):
            # resize the token embeddings
            new_embedding.weight.data[token_id] = tensor.to(
                new_embedding.weight.device, dtype
            )


def match_encoder_to_tokenizer(tokenizer, text_encoder):
    tokenizer_len = len(tokenizer)
    orig_embedding = text_encoder.get_input_embeddings()

    # If there's a hook attached, reduce/remove any re-application
    if (hook := get_hook(orig_embedding, ApplyEmbeddingHook)) is not False:
        hook.additions = {k: v for k, v in hook.additions.items() if k < tokenizer_len}
        if not hook.additions:
            remove_hook(orig_embedding, ApplyEmbeddingHook)

    # And if we're not on meta right now, fix that version too
    if orig_embedding.weight.device != torch.device("meta"):
        # Resize the token embeddings
        new_embedding = text_encoder.resize_token_embeddings(tokenizer_len)
        # Transfer any hooks
        if new_embedding is not orig_embedding and is_hooked(orig_embedding):
            add_hook(new_embedding, orig_embedding._hf_hook, replace=True)
