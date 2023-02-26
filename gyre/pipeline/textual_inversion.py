import logging

logger = logging.getLogger(__name__)


def apply_ti_token(pipe, token, tensor):
    text_encoder = pipe.text_encoder
    tokenizer = pipe.tokenizer

    # cast to dtype of text_encoder
    dtype = text_encoder.get_input_embeddings().weight.dtype

    if tokenizer.add_tokens(token) == 0:
        logger.debug(
            f"Tokenizer already contains the token {token} when loading textual inversion"
        )

    # resize the token embeddings
    text_encoder.resize_token_embeddings(len(tokenizer))

    # get the id for the token and assign the embeds
    token_id = tokenizer.convert_tokens_to_ids(token)
    text_encoder.get_input_embeddings().weight.data[token_id] = tensor.to(dtype)
