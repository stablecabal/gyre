# From https://github.com/ewrfcas/ZITS-PlusPlus/blob/main/test.py
# Distributed under Apache-2.0
# Changes:
# - None

import cv2
import numpy as np

ones_filter = np.ones((3, 3), dtype=np.float32)
d_filter1 = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]], dtype=np.float32)
d_filter2 = np.array([[0, 0, 0], [1, 1, 0], [1, 1, 0]], dtype=np.float32)
d_filter3 = np.array([[0, 1, 1], [0, 1, 1], [0, 0, 0]], dtype=np.float32)
d_filter4 = np.array([[0, 0, 0], [0, 1, 1], [0, 1, 1]], dtype=np.float32)


def load_masked_position_encoding(mask):
    ori_mask = mask.copy()
    ori_h, ori_w = ori_mask.shape[0:2]
    ori_mask = ori_mask / 255
    mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_AREA)
    mask[mask > 0] = 255
    h, w = mask.shape[0:2]
    mask3 = mask.copy()
    mask3 = 1.0 - (mask3 / 255.0)
    pos = np.zeros((h, w), dtype=np.int32)
    direct = np.zeros((h, w, 4), dtype=np.int32)
    i = 0
    while np.sum(1 - mask3) > 0:
        i += 1
        mask3_ = cv2.filter2D(mask3, -1, ones_filter)
        mask3_[mask3_ > 0] = 1
        sub_mask = mask3_ - mask3
        pos[sub_mask == 1] = i

        m = cv2.filter2D(mask3, -1, d_filter1)
        m[m > 0] = 1
        m = m - mask3
        direct[m == 1, 0] = 1

        m = cv2.filter2D(mask3, -1, d_filter2)
        m[m > 0] = 1
        m = m - mask3
        direct[m == 1, 1] = 1

        m = cv2.filter2D(mask3, -1, d_filter3)
        m[m > 0] = 1
        m = m - mask3
        direct[m == 1, 2] = 1

        m = cv2.filter2D(mask3, -1, d_filter4)
        m[m > 0] = 1
        m = m - mask3
        direct[m == 1, 3] = 1

        mask3 = mask3_

    abs_pos = pos.copy()
    rel_pos = pos / (256 / 2)  # to 0~1 maybe larger than 1
    rel_pos = (rel_pos * 128).astype(np.int32)
    rel_pos = np.clip(rel_pos, 0, 128 - 1)

    if ori_w != w or ori_h != h:
        rel_pos = cv2.resize(rel_pos, (ori_w, ori_h), interpolation=cv2.INTER_NEAREST)
        rel_pos[ori_mask == 0] = 0
        direct = cv2.resize(direct, (ori_w, ori_h), interpolation=cv2.INTER_NEAREST)
        direct[ori_mask == 0, :] = 0

    return rel_pos, abs_pos, direct
