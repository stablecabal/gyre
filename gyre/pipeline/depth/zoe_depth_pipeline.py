import torch

from gyre import images
from gyre.pipeline.depth.zoe_model_wrapper import ZoeModelWrapper


class ZoeDepthPipeline:
    zoe_depth_estimator: ZoeModelWrapper

    def __init__(self, zoe_depth_estimator):
        self.zoe_depth_estimator = zoe_depth_estimator

    def to(self, device):
        self.zoe_depth_estimator.to(device)

    def pipeline_modules(self):
        return [("zoe_depth_estimator", self.zoe_depth_estimator)]

    @torch.no_grad()
    def __call__(self, tensor, normalise=True, invert=True):
        sample = tensor

        # Get device and dtype of model
        device = self.zoe_depth_estimator.device
        dtype = self.zoe_depth_estimator.dtype

        # CHW in RGB only (strip batch, strip A)
        sample = images.normalise_tensor(tensor).to(device, dtype)

        # And predict
        depth_map = self.zoe_depth_estimator.infer(sample)

        # _model output is a single monochrome 1HW. Convert to B
        # Normalise
        if normalise:
            depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
            depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
            depth_map = (depth_map - depth_min) / (depth_max - depth_min)

        if invert:
            depth_map = depth_map.max() - depth_map

        return depth_map.to(tensor)
