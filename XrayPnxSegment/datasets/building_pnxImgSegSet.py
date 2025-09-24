"""
 SPDX-License-Identifier: MIT
 Copyright (c) 2025, yasaisen (clover)
 
 This file is part of a project licensed under the MIT License.
 See the LICENSE file in the project root for more information.
 
 last modified in 2506222348
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import random


def show_stats(name, data):
    print(f'== {name} ==')
    print('Ratio of pneumothorax:')
    print(data['has_pnx'].value_counts(normalize=True))
    print('\nratio distribution:')
    print(data[data['has_pnx']]['ratio'].describe())
    print('-'*40)

def sample_subset(
    data_list, 
    subset_sample_ratio, 
):
    df = pd.DataFrame(data_list)
    
    df['cls_prob'] = pd.to_numeric(df['cls_prob'], errors='coerce')  # 轉 float，invalid 變 NaN
    df = df.dropna(subset=['cls_prob'])
    df['hard_weight'] = 1.0
    df.loc[(~df['has_pnx']) & (df['cls_prob'] > 0.5), 'hard_weight'] = 2.0
    df.loc[df['has_pnx'] & (df['cls_prob'] < 0.5), 'hard_weight'] = 1.5



    bins = [-1e-9, 0, 0.01, 0.05, 1]
    labels = ['none', 'small', 'medium', 'large']
    df['ratio_bin'] = pd.cut(df['ratio'], bins=bins, labels=labels, right=False)

    # set all of 'df['ratio_bin'] are none, then process df['has_pnx'] with qcut (q1, q2, q3, q4)
    df['ratio_bin'] = 'none'
    mask = df['has_pnx']
    df.loc[mask, 'ratio_bin'] = pd.qcut(df.loc[mask, 'ratio'], q=4, labels=['q1','q2','q3','q4'])

    df['strata'] = df['has_pnx'].astype(int).astype(str) + '_' + df['ratio_bin'].astype(str)
    print(f'df: {df.head()}')
    print(f'len: {len(df)}')
    # # fuck
    # rest_df, inuse_df = train_test_split(
    #     df,
    #     test_size=subset_sample_ratio,
    #     stratify=df['strata'],
    #     random_state=42,
    # )
    inuse_df = df

    print(df['strata'].value_counts(normalize=True).head())
    print(inuse_df['strata'].value_counts(normalize=True).head())

    show_stats('original', df)
    show_stats('inuse_df', inuse_df)

    return inuse_df.to_dict(orient='records')

def crop_mask_to_square(
    mask: np.ndarray,
    pad: int = 0,
    min_size: int | None = None
):
    assert mask.ndim == 2, "mask must be H×W"

    ys, xs = np.where(mask > 0)
    h, w = mask.shape

    if len(xs) == 0:
        side = min(h, w) if min_size is None else min_size
        side = min(side, min(h, w))
        top = (h - side) // 2
        left = (w - side) // 2
        bottom, right = top + side, left + side
    else:
        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()

        # grow the box
        y0, x0 = max(y0 - pad, 0), max(x0 - pad, 0)
        y1, x1 = min(y1 + pad, h - 1), min(x1 + pad, w - 1)

        box_h, box_w = y1 - y0 + 1, x1 - x0 + 1
        side = max(box_h, box_w)
        if min_size:
            side = max(side, min_size)

        cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
        half = side // 2
        top, bottom = cy - half, cy - half + side
        left, right = cx - half, cx - half + side

        # shift if we ran out of image
        dy1 = max(0, -top)
        dx1 = max(0, -left)
        dy2 = max(0, bottom - h)
        dx2 = max(0, right - w)
        top += dy1 - dy2
        bottom += dy1 - dy2
        left += dx1 - dx2
        right += dx1 - dx2

        top, left = int(top), int(left)
        bottom, right = int(bottom), int(right)

    cropped = mask[top:bottom, left:right]

    ch, cw = cropped.shape
    if ch != cw:
        side = max(ch, cw)
        pad_y = (side - ch) // 2
        pad_x = (side - cw) // 2
        square = np.zeros((side, side), dtype=mask.dtype)
        square[pad_y:pad_y + ch, pad_x:pad_x + cw] = cropped
        cropped = square
        return cropped, (pad_y, pad_y + ch, pad_x, pad_x + cw)
    
    return cropped, (top, bottom, left, right)

def random_square_crop(
    image: np.ndarray, 
    crop_size: int
) -> np.ndarray:
    h, w = image.shape[:2]

    if crop_size > min(h, w):
        raise ValueError("The crop size cannot be larger than the original image size")

    x_start = random.randint(0, w - crop_size)
    y_start = random.randint(0, h - crop_size)

    cropped = image[y_start:y_start + crop_size, x_start:x_start + crop_size]
    return cropped, (y_start, y_start + crop_size, x_start, x_start + crop_size)