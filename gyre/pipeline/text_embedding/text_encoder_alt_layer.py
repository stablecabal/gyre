from typing import Literal

from gyre.pipeline.prompt_types import ClipLayer, normalise_clip_layer


class TextEncoderAltLayer:
    def __init__(
        self,
        text_encoder,
        layer: ClipLayer = "final",
    ):
        self.text_encoder = text_encoder
        self.layer: ClipLayer = layer

    def __call__(self, input_ids):
        layer = normalise_clip_layer(self.layer, "final")

        text_embeddings = self.text_encoder(
            input_ids,
            output_hidden_states=(layer != "final"),
            return_dict=True,
        )

        if layer == "final":
            res = text_embeddings.last_hidden_state
        elif layer == "penultimate":
            res = self.text_encoder.text_model.final_layer_norm(
                text_embeddings.hidden_states[-2]
            )
        else:
            res = self.text_encoder.text_model.final_layer_norm(
                text_embeddings.hidden_states[-layer]
            )

        # text_encoder clients expect tuple of (final layer, pool)
        return (res, None)
