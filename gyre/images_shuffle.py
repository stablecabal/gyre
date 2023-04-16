# Based on https://github.com/lllyasviel/ControlNet-v1-1-nightly/blob/main/annotator/shuffle/__init__.py
# Distributed under Apache-2.0

# Changes:
# - Combined functions from two files

import random

import cv2
import numpy as np


def make_noise_disk(H, W, C, F):
    noise = np.random.uniform(low=0, high=1, size=((H // F) + 2, (W // F) + 2, C))
    noise = cv2.resize(noise, (W + 2 * F, H + 2 * F), interpolation=cv2.INTER_CUBIC)
    noise = noise[F : F + H, F : F + W]
    noise -= np.min(noise)
    noise /= np.max(noise)
    if C == 1:
        noise = noise[:, :, None]
    return noise


class ContentShuffleDetector:
    def __call__(self, img, h=None, w=None, f=None):
        H, W, C = img.shape
        if h is None:
            h = H
        if w is None:
            w = W
        if f is None:
            f = 256
        x = make_noise_disk(h, w, 1, f) * float(W - 1)
        y = make_noise_disk(h, w, 1, f) * float(H - 1)
        flow = np.concatenate([x, y], axis=2).astype(np.float32)
        return cv2.remap(img, flow, None, cv2.INTER_LINEAR)


class ColorShuffleDetector:
    def __call__(self, img):
        H, W, C = img.shape
        F = random.randint(64, 384)
        A = make_noise_disk(H, W, 3, F)
        B = make_noise_disk(H, W, 3, F)
        C = (A + B) / 2.0
        A = (C + (A - C) * 3.0).clip(0, 1)
        B = (C + (B - C) * 3.0).clip(0, 1)
        L = img.astype(np.float32) / 255.0
        Y = A * L + B * (1 - L)
        Y -= np.min(Y, axis=(0, 1), keepdims=True)
        Y /= np.maximum(np.max(Y, axis=(0, 1), keepdims=True), 1e-5)
        Y *= 255.0
        return Y.clip(0, 255).astype(np.uint8)
