import math
from contextlib import contextmanager
from typing import Literal

import cv2
import numpy as np
import torch
from mmdet.apis import inference_detector
from mmpose.apis import inference_top_down_pose_model, process_mmdet_results

from gyre import images

# These keypose arrays and function derived from
# https://github.com/TencentARC/T2I-Adapter/blob/main/ldm/modules/extra_condition/utils.py
keypose_skeleton = [
    [15, 13],
    [13, 11],
    [16, 14],
    [14, 12],
    [11, 12],
    [5, 11],
    [6, 12],
    [5, 6],
    [5, 7],
    [6, 8],
    [7, 9],
    [8, 10],
    [1, 2],
    [0, 1],
    [0, 2],
    [1, 3],
    [2, 4],
    [3, 5],
    [4, 6],
]

keypose_kpt_color = [
    [51, 153, 255],
    [51, 153, 255],
    [51, 153, 255],
    [51, 153, 255],
    [51, 153, 255],
    [0, 255, 0],
    [255, 128, 0],
    [0, 255, 0],
    [255, 128, 0],
    [0, 255, 0],
    [255, 128, 0],
    [0, 255, 0],
    [255, 128, 0],
    [0, 255, 0],
    [255, 128, 0],
    [0, 255, 0],
    [255, 128, 0],
]

keypose_link_color = [
    [0, 255, 0],
    [0, 255, 0],
    [255, 128, 0],
    [255, 128, 0],
    [51, 153, 255],
    [51, 153, 255],
    [51, 153, 255],
    [51, 153, 255],
    [0, 255, 0],
    [255, 128, 0],
    [0, 255, 0],
    [255, 128, 0],
    [51, 153, 255],
    [51, 153, 255],
    [51, 153, 255],
    [51, 153, 255],
    [51, 153, 255],
    [51, 153, 255],
    [51, 153, 255],
]


def exclude_link(p1, p2, iw, ih, thresh):
    offs = [round(off) for off in p1[:2].tolist() + p2[:2].tolist()] + [p1[2], p2[2]]
    rngs = [(0, iw), (0, ih), (0, iw), (0, ih), (thresh, math.inf), (thresh, math.inf)]

    for off, rng in zip(offs, rngs):
        if off < rng[0] or off > rng[1]:
            return True

    return False


def render_keypose(img, pose_result, score_thr=0.1, radius=2, thickness=2):
    """Draw keypoints and links on an image.
    Args:
            img (ndarry): The image to draw poses on.
            pose_result (list[kpts]): The poses to draw. Each element kpts is
                a set of K keypoints as an Kx3 numpy.ndarray, where each
                keypoint is represented as x, y, score.
            score_thr (float, optional): Minimum score of keypoints
                to be shown. Default: 0.3.
            thickness (int): Thickness of lines.
    """

    img_h, img_w, _ = img.shape
    img = np.zeros(img.shape)

    for idx, kpts in enumerate(pose_result):
        kpts = kpts["keypoints"]
        kpts = np.array(kpts, copy=False)

        # draw each point on image
        assert len(keypose_kpt_color) == len(kpts)

        for kid, kpt in enumerate(kpts):
            x_coord, y_coord, kpt_score = int(kpt[0]), int(kpt[1]), kpt[2]

            if kpt_score < score_thr or keypose_kpt_color[kid] is None:
                # skip the point that should not be drawn
                continue

            color = tuple(int(c) for c in keypose_kpt_color[kid])
            cv2.circle(img, (int(x_coord), int(y_coord)), radius, color, -1)

        # draw links

        for sk_id, sk in enumerate(keypose_skeleton):
            if exclude_link(kpts[sk[0]], kpts[sk[1]], img_w, img_h, score_thr):
                continue

            pos1 = (round(kpts[sk[0], 0]), round(kpts[sk[0], 1]))
            pos2 = (round(kpts[sk[1], 0]), round(kpts[sk[1], 1]))
            color = tuple(int(c) for c in keypose_link_color[sk_id])

            cv2.line(img, pos1, pos2, color, thickness=thickness)

    return img


def keypose_to_openpose(points):

    # Add the additional point in the middle of the should blade (an approximation)
    points = np.append(points, [(points[5] + points[6]) / 2], axis=0)

    remap = [0, 17, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]
    return np.array([points[i] for i in remap])


openpose_colors = [
    [255, 0, 0],
    [255, 85, 0],
    [255, 170, 0],
    [255, 255, 0],
    [170, 255, 0],
    [85, 255, 0],
    [0, 255, 0],
    [0, 255, 85],
    [0, 255, 170],
    [0, 255, 255],
    [0, 170, 255],
    [0, 85, 255],
    [0, 0, 255],
    [85, 0, 255],
    [170, 0, 255],
    [255, 0, 255],
    [255, 0, 170],
    [255, 0, 85],
]

openpose_skeleton = [
    [1, 2],
    [1, 5],
    [2, 3],
    [3, 4],
    [5, 6],
    [6, 7],
    [1, 8],
    [8, 9],
    [9, 10],
    [1, 11],
    [11, 12],
    [12, 13],
    [1, 0],
    [0, 14],
    [14, 16],
    [0, 15],
    [15, 17],
]


@contextmanager
def blend(img, alpha):
    work = img.copy()
    yield work
    cv2.addWeighted(img, 1 - alpha, work, alpha, 0, dst=img)


def render_openpose(img, pose_result, score_thr=0.1):
    img_h, img_w, _ = img.shape
    img = np.zeros(img.shape)

    colors = [[x[2], x[1], x[0]] for x in openpose_colors]

    for pose_i, pose in enumerate(pose_result):
        points = keypose_to_openpose(pose["keypoints"])

        for i, seq in enumerate(openpose_skeleton):
            if exclude_link(points[seq[0]], points[seq[1]], img_w, img_h, score_thr):
                continue

            a = points[seq[0]][:2]
            b = points[seq[1]][:2]

            # Midpoint
            m = (a + b) / 2
            # Line segment length
            sq = (a - b) ** 2
            length = (sq[0] + sq[1]) ** 0.5
            # X radius of arc
            rad = length / 2
            # X angle of arc
            ang = math.degrees(math.atan2(a[1] - b[1], a[0] - b[0]))

            with blend(img, 0.5) as work:
                polygon = cv2.ellipse2Poly(
                    (round(m[0]), round(m[1])), (round(rad), 4), round(ang), 0, 360, 1
                )
                cv2.fillConvexPoly(work, polygon, colors[i])

        for i, point in enumerate(points):
            if point[2] < score_thr:
                continue

            p = tuple((round(x.item()) for x in point[:2]))

            with blend(img, 0.5) as work:
                cv2.circle(work, p, 5, colors[i], thickness=-1)

    return img


class MmposePipeline:
    def __init__(self, detector, posemodel):
        self.detector = detector
        self.posemodel = posemodel

    def to(self, device):
        self.detector.to(device)
        self.posemodel.to(device)

    def pipeline_modules(self):
        return [("detector", self.detector), ("posemodel", self.posemodel)]

    @torch.no_grad()
    def __call__(
        self, tensor, output_format: Literal["keypose", "openpose"] = "keypose"
    ):
        if tensor.ndim != 4 or tensor.shape[2] < 3:
            raise ValueError("Tensor must be RGB image in BCHW format")

        samples = [image for image in images.toCV(tensor[:, [0, 1, 2]])]

        outputs = []
        for sample in samples:
            mmdet_results = inference_detector(self.detector, sample)
            person_results = process_mmdet_results(mmdet_results, 1)

            dataset = self.posemodel.cfg.data["test"]["type"]
            return_heatmap = False
            output_layer_names = None

            pose_results, returned_outputs = inference_top_down_pose_model(
                self.posemodel,
                sample,
                person_results,
                bbox_thr=0.2,
                format="xyxy",
                dataset=dataset,
                dataset_info=None,
                return_heatmap=return_heatmap,
                outputs=output_layer_names,
            )

            # show the results
            if output_format == "keypose":
                pose = render_keypose(sample, pose_results, score_thr=0.4)
            else:
                pose = render_openpose(sample, pose_results, score_thr=0.4)

            outputs.append(images.fromCV(pose))

        return torch.cat(outputs, dim=0).to(tensor.device, tensor.dtype)
