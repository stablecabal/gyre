import logging
import kornia
import torch
import torchvision

from gyre.logging import VisualRecord as vr
from gyre import images
from diffusers.models import modeling_utils

from .zitspp_postition_encoder import load_masked_position_encoding
from .zitspp_nms import get_nms as get_np_nms  # This is a pure pytorch version

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import cv2
import numpy as np

import torch.nn.functional as FF
import torchvision.transforms.functional as F

# A lot of this file is derived from https://github.com/ewrfcas/ZITS-PlusPlus/blob/main/test.py
# (particularly resize, to_tensor, and __call__)
# and https://github.com/ewrfcas/ZITS-PlusPlus/blob/main/trainers/pl_trainers.py
# (particularly to_device and lsm_hasp_inference)
#
# Used under Apache-2.0
#
# Changes:
# - Subsetted and merged
# - Extensive rewriting to fit pipeline structure, change inputs and avoid new dependancies


def resize(img, height, width, center_crop=False):
    imgh, imgw = img.shape[0:2]

    if center_crop and imgh != imgw:
        # center crop
        side = np.minimum(imgh, imgw)
        j = (imgh - side) // 2
        i = (imgw - side) // 2
        img = img[j : j + side, i : i + side, ...]

    if imgh > height and imgw > width:
        inter = cv2.INTER_AREA
    else:
        inter = cv2.INTER_LINEAR
    img = cv2.resize(img, (width, height), interpolation=inter)

    return img


# Converts an RGB CV in HWC (0..1 float or 0..255 byte) to a RGB Tensor in CHW 0..1
def to_tensor(img, norm=False):
    # img = Image.fromarray(img)
    img_t = F.to_tensor(img).float()
    if norm:
        img_t = F.normalize(img_t, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    return img_t


# Converts an RGB Tensor in CHW 0..1 to an RGB CV in HWC (configurable max and type)
def to_cv(img, max=1.0, nptype=None):
    result = img.permute(1, 2, 0).cpu().numpy() * max
    if nptype is not None:
        result = result.astype(nptype)
    return result


def to_device(data, device):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    if isinstance(data, dict):
        for key in data:
            if isinstance(data[key], torch.Tensor):
                data[key] = data[key].to(device)
        return data
    if isinstance(data, list):
        return [to_device(d, device) for d in data]


# --- Matches skimage.rgb2gray

RGB2GREY_COEFF = torch.tensor([0.2125, 0.7154, 0.0721], dtype=torch.float64)


def rgb2gray(input):
    if isinstance(input, np.ndarray):
        input = to_tensor(input.copy())

    input = input.to(torch.float64)
    result = kornia.color.rgb_to_grayscale(input, RGB2GREY_COEFF)
    return to_cv(result)
    # return result.permute(1, 2, 0).cpu().numpy()


class ZitsPPPipeline:
    module: torch.nn.Module

    def __init__(self, module, wf):
        self.structure_upsample = module.structure_upsample
        self.edgeline_tsr = module.edgeline_tsr
        self.ftr = module.ftr
        self.wf = wf

    def to(self, device):
        self.module.to(device)

    def pipeline_modules(self):
        return [
            ("structure_upsample", self.structure_upsample),
            ("edgeline_tsr", self.edgeline_tsr),
            ("ftr", self.ftr),
            ("wf", self.wf),
        ]

    @torch.no_grad()
    def lsm_hasp_inference(self, image, mask, output_size=(256, 256), mask_th=0.925):
        LCNN_MEAN = [109.730, 103.832, 98.681]
        LCNN_STD = [22.275, 22.124, 23.229]
        INPUT_SIZE = (512, 512)
        w, h = output_size

        # Image should be mid-grey in the mask
        masked_image = image * (1 - mask) + torch.ones_like(image) * mask * 0.5
        # Image should be strictly INPUT_SIZE
        masked_image = images.rescale(masked_image, *INPUT_SIZE, fit="strict")
        # Image should be 0..255
        masked_image = masked_image * 255.0
        # Image should be normalized
        masked_image = F.normalize(masked_image, mean=LCNN_MEAN, std=LCNN_STD)

        device = modeling_utils.get_parameter_device(self.wf)
        dtype = modeling_utils.get_parameter_dtype(self.wf)

        lines_tensor = []
        for i in range(masked_image.shape[0]):
            lmap = np.zeros((*output_size, 1), dtype="uint8")

            input_image = masked_image[[i]].to(device, dtype)
            output_masked = self.wf(input_image)
            output_masked = to_device(output_masked, "cpu")

            if output_masked["num_proposals"] == 0:
                lines_masked = []
                scores_masked = []
            else:
                lines_masked = output_masked["lines_pred"].numpy()
                lines_masked = [
                    [line[0] * w, line[1] * h, line[2] * w, line[3] * h]
                    for line in lines_masked
                ]
                lines_masked = [list(map(int, line)) for line in lines_masked]

                scores_masked = output_masked["lines_score"].numpy()

            for line, score in zip(lines_masked, scores_masked):
                if score > mask_th:
                    temp = np.zeros_like(lmap)
                    cv2.line(temp, line[0:2], line[2:4], (255), 1, cv2.LINE_AA)
                    lmap = np.maximum(lmap, temp)

            lines_tensor.append(images.fromCV(lmap))

        lines_tensor = torch.cat(lines_tensor, dim=0)

        logger.debug(
            vr(
                "LSM HASP {image} {mask} {lines}",
                image=image,
                mask=mask,
                lines=lines_tensor,
            )
        )

        return lines_tensor.detach().to(image)

    @torch.no_grad()
    def __call__(self, image, mask, obj_removal=False):
        image = images.normalise_tensor(image, 3)
        mask = images.normalise_tensor(mask, 1)

        # Harden mask
        mask[mask >= 0.999] = 1
        mask[mask < 1] = 0

        # Calculate 256x256 versions of image and mask
        image_256 = images.rescale(image, 256, 256, "strict", sharpness=0)
        mask_256 = images.rescale(mask, 256, 256, "strict", sharpness=0)
        mask_256[mask_256 >= 0.001] = 1
        mask_256[mask_256 < 1] = 0

        # Perform line detection
        line_256 = self.lsm_hasp_inference(image, mask, mask_th=0.85)

        # Convert to CV
        img = images.toCV(image)[0]
        img = img[:, :, ::-1].copy()  # ZitsPP works in RGB mode

        img_256 = images.toCV(image_256)[0]
        img_256 = img_256[:, :, ::-1].copy()  # ZitsPP works in RGB mode

        mask = images.toCV(mask)[0]
        mask = (mask > 0).astype(np.uint8) * 255

        mask_256 = images.toCV(mask_256)[0]
        mask_256 = (mask_256 > 0).astype(np.uint8) * 255

        # resize/crop if needed
        imgh, imgw, _ = img.shape

        mask = mask[:, :, 0]
        mask_256 = mask_256[:, :, 0]

        # load gradient
        img_gray = rgb2gray(img_256) * 255
        sobelx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0).astype(np.float)
        sobely = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1).astype(np.float)

        logger.debug(vr("Input {rgb} {gray}", rgb=img_256, gray=img_gray))
        logger.debug(vr("Sobel {sobelx} {sobely}", sobelx=sobelx, sobely=sobely))

        img_gray = rgb2gray(img) * 255
        sobelx_hr = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0).astype(np.float)
        sobely_hr = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1).astype(np.float)

        logger.debug(vr("HR Input {rgb} {gray}", rgb=img.copy(), gray=img_gray))
        logger.debug(
            vr("HR Sobel {sobelx} {sobely}", sobelx=sobelx_hr, sobely=sobely_hr)
        )

        batch = dict()
        batch["image"] = to_tensor(img, norm=True)
        batch["img_256"] = to_tensor(img_256, norm=True)
        batch["mask"] = to_tensor(mask)
        batch["mask_256"] = to_tensor(mask_256)
        batch["gradientx"] = torch.from_numpy(sobelx).unsqueeze(0).float()
        batch["gradienty"] = torch.from_numpy(sobely).unsqueeze(0).float()

        batch["gradientx_hr"] = torch.from_numpy(sobelx_hr).unsqueeze(0).float()
        batch["gradienty_hr"] = torch.from_numpy(sobely_hr).unsqueeze(0).float()

        batch["line_256"] = line_256[0]
        batch["size_ratio"] = -1

        # # load pos encoding
        rel_pos, abs_pos, direct = load_masked_position_encoding(mask)
        batch["rel_pos"] = torch.LongTensor(rel_pos)
        batch["abs_pos"] = torch.LongTensor(abs_pos)
        batch["direct"] = torch.LongTensor(direct)

        batch["H"] = -1

        device = modeling_utils.get_parameter_device(self.ftr)
        dtype = modeling_utils.get_parameter_dtype(self.ftr)
        for k in batch:
            if type(batch[k]) is torch.Tensor:
                batch[k] = batch[k].to(device).unsqueeze(0)

        # inapint prior
        edge_pred, line_pred = self.edgeline_tsr.forward(
            batch["img_256"], batch["line_256"], masks=batch["mask_256"]
        )
        line_pred = (
            batch["line_256"] * (1 - batch["mask_256"]) + line_pred * batch["mask_256"]
        )

        edge_pred = edge_pred.detach()
        line_pred = line_pred.detach()

        logger.debug(
            vr(
                "Predicitions {edge_pred} {line_pred}",
                edge_pred=edge_pred,
                line_pred=line_pred,
            )
        )

        current_size = 256
        while current_size * 2 <= max(imgh, imgw):
            # nms for HR
            line_pred = self.structure_upsample(line_pred)[0]
            line_pred = torch.sigmoid((line_pred + 2) * 2)
            edge_pred_nms = get_np_nms(edge_pred, binary_threshold=50)
            edge_pred_nms = self.structure_upsample(edge_pred_nms)[0]
            edge_pred_nms = torch.sigmoid((edge_pred_nms + 2) * 2)
            current_size *= 2

        edge_pred_nms = images.rescale(
            edge_pred_nms, width=imgw, height=imgh, fit="strict"
        )
        edge_pred = images.rescale(edge_pred, width=imgw, height=imgh, fit="strict")
        edge_pred[edge_pred >= 0.25] = edge_pred_nms[edge_pred >= 0.25]
        line_pred = images.rescale(line_pred, width=imgw, height=imgh, fit="strict")

        logger.debug(
            vr(
                "Upscaled predicitions {edge_pred} {line_pred}",
                edge_pred=edge_pred,
                line_pred=line_pred,
            )
        )

        batch["edge"] = edge_pred.detach()
        batch["line"] = line_pred.detach()

        prediction = self.ftr.forward(batch)
        gen_ema_img = batch["image"] * (1 - batch["mask"]) + prediction * batch["mask"]

        prediction = torch.clamp(prediction, -1, 1)
        prediction = (prediction + 1) / 2

        gen_ema_img = torch.clamp(gen_ema_img, -1, 1)
        gen_ema_img = (gen_ema_img + 1) / 2

        logger.debug(
            vr(
                "Result {prediction} {gen_ema_img}",
                prediction=prediction,
                gen_ema_img=gen_ema_img,
            )
        )

        return gen_ema_img.to(image)
