import logging

logger = logging.getLogger(__name__)


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


def apply_multidim_ti_token(tokenizer, text_encoder, token, tensor):
    subtokens = []

    for i in range(0, tensor.shape[0]):
        subtoken = f"{token}-{i}"
        apply_ti_token(tokenizer, text_encoder, subtoken, tensor[i])
        subtokens.append(subtoken)

    attach_multidim_handler(tokenizer)
    tokenizer._multidim_mappings[token] = " ".join(subtokens)


def apply_ti_token(tokenizer, text_encoder, token, tensor):
    # cast to dtype of text_encoder
    dtype = text_encoder.get_input_embeddings().weight.dtype

    if len(tensor.shape) == 2:
        logger.debug("Multi-dimensional")
        apply_multidim_ti_token(tokenizer, text_encoder, token, tensor)

    else:
        if tokenizer.add_tokens(token) == 0:
            logger.debug(
                f"Tokenizer already contains the token {token} when loading textual inversion"
            )

        # resize the token embeddings
        text_encoder.resize_token_embeddings(len(tokenizer))

        # get the id for the token and assign the embeds
        token_id = tokenizer.convert_tokens_to_ids(token)
        text_encoder.get_input_embeddings().weight.data[token_id] = tensor.to(dtype)
