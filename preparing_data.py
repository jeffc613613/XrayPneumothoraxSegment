"""
 SPDX-License-Identifier: MIT
 Copyright (c) 2025, yasaisen (clover)
 
 This file is part of a project licensed under the MIT License.
 See the LICENSE file in the project root for more information.
 
 last modified in 2506222348
"""

import numpy as np
import random
import json
import os
from tqdm import tqdm
import cv2

from XrayPnxSegment.common.utils import convert
from XrayPnxSegment.datasets.building_pnxImgSegSet import (
    sample_subset, 
    crop_mask_to_square, 
    random_square_crop
)
from pneumothorax_predictor import predict_pneumothorax


ROOT_PATH = os.getcwd()
# ROOT_PATH = '/home/yasaisen/Desktop/250610'
RATIO = 0.9  # Subset sample ratio

model_path = "res18_pneumothorax_classifier.pth"

def main():
    data_path = os.path.join(ROOT_PATH, 'siim-acr-pneumothorax', 'png_images')
    cropped_path = data_path.replace('png_images', 'cropped_masks')
    os.makedirs(cropped_path, exist_ok=True)

    error_list = []
    data_list = []
    dir_list = os.listdir(data_path)
    for idx in tqdm(range(len(dir_list))):
        try:
            img_path = os.path.join(data_path, dir_list[idx])
            msk_path = img_path.replace('png_images', 'png_masks')

            # img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            cls_prob = predict_pneumothorax(model_path, img_path)
            cls_prob = f"{cls_prob:.4f}"
            # print(f"氣胸機率 (via function): {prob2:.4f}")

            msk = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE)
            num_zeros = int(np.count_nonzero(msk == 0))
            num_ones = int(np.count_nonzero(msk == 255))

            if 'train' in img_path:
                split = 'train'
            elif 'test' in img_path:
                split = 'test'

            if num_ones > 0:
                cropped, coords = crop_mask_to_square(msk, pad=128, min_size=128)
                msk = np.stack([cropped, cropped, cropped], axis=-1)
            else:
                cropped, coords = random_square_crop(msk, random.randint(128, int(msk.shape[0])))
                msk = np.stack([cropped, cropped, cropped], axis=-1)

            filename = 'cropped_' + msk_path.split("\\")[-1]
            full_filename = os.path.join(cropped_path, filename)
            cv2.imwrite(full_filename, msk)
            # print(os.path.join(cropped_path, filename))

            data_list += [{
                'idx': idx, 
                'image_path': img_path, 
                'mask_path': msk_path, 
                'has_pnx': num_ones > 0,
                'ratio': num_ones / num_zeros, 
                'cropped_mask_path': full_filename,
                'coords': coords,
                'split': split,
                'cls_prob': cls_prob
            }]

        except Exception as e:
            error_list += [{
                'idx': idx, 
                'error': e
            }]
    print('data_list', len(data_list))
    print('error_list', len(error_list))

    train_data_list = []
    for sample in data_list:
        if sample['split'] == 'train':
            train_data_list += [sample]
    proed_train_data_list = sample_subset(
        data_list=train_data_list, 
        subset_sample_ratio=RATIO
    )

    test_data_list = []
    for sample in data_list:
        if sample['split'] == 'test':
            test_data_list += [sample]
    proed_test_data_list = sample_subset(
        data_list=test_data_list, 
        subset_sample_ratio=RATIO
    )

    # train_data_list = []
    # for sample in data_list:
    #     if sample['split'] == 'train':
    #         train_data_list += [sample]

    # test_data_list = []
    # for sample in data_list:
    #     if sample['split'] == 'test':
    #         test_data_list += [sample]

    # if RATIO >= 1.0:
    #     proed_train_data_list = train_data_list
    #     proed_test_data_list = test_data_list
    # else:
    #     print(f"flag1")
    #     proed_train_data_list = sample_subset(
    #         data_list=train_data_list, 
    #         subset_sample_ratio=RATIO
    #     )
    #     proed_test_data_list = sample_subset(
    #         data_list=test_data_list, 
    #         subset_sample_ratio=RATIO
    #     )

    proed_data_list = proed_train_data_list + proed_test_data_list

    file_save_path = 'subset_data_2508240320.json'
    with open(file_save_path, "w") as file:
        json.dump(proed_data_list, file, indent=4, default=convert)

if __name__ == "__main__":
    main()













