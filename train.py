"""
 SPDX-License-Identifier: MIT
 Copyright (c) 2025, yasaisen (clover)
 
 This file is part of a project licensed under the MIT License.
 See the LICENSE file in the project root for more information.
 
 last modified in 2506222348
"""

import torch
import os
import numpy as np
import time
import json
from datetime import datetime
from torch.utils.data import DataLoader, WeightedRandomSampler, SubsetRandomSampler
from sklearn.utils.class_weight import compute_class_weight
from XrayPnxSegment.common.utils import plot_training_comparison
from XrayPnxSegment.datasets.pnxImgSegSet import pnxImgSegSet, validate_dataset
from XrayPnxSegment.models.modeling_segModels import get_DeepLabV3Plus, get_Unet, get_FPN_ResNet50,get_PSPNet_ResNet50
from XrayPnxSegment.processors.img_processor import get_transform
from XrayPnxSegment.trainer.building_SegModelTrainer import get_lossFunc, get_optim, train_model
from collections import defaultdict

# save_time_stamp = os.path.join(os.getcwd(), 'checkpoints', datetime.now().strftime("%y%m%d%H%M"))
# config = {
#         'bsz': 12,
#         # 'lr': 1e-4,
#         # 'num_epoch': 150,
#         # 'img_size': (768, 768),
#         'mask_key': 'cropped_mask_path',  # 'mask_path', 'cropped_mask_path'
#         'skip_has_pnx': False,
#         'calc_class_weights': True,
#         'criterion': 'combined',          # 'BCE', 'combined'
#         'root_path': os.getcwd(),
#         'meta_path': 'subset_data_2508132253_small.json',
#         'save_path': os.path.join(os.getcwd(), 'checkpoints', save_time_stamp),
#         'device': 'cuda' if torch.cuda.is_available() else 'cpu',
#     }
 

# ---------- Sampler ----------
def create_ratio_based_sampler(
    dataset, 
    target_pos_ratio, 
    total_samples,
    strata_weights=None
):
    """
    Creates a sampler that maintains a specific ratio of positive (pneumothorax) 
    to negative (non-pneumothorax) samples in each batch/epoch, with support for 
    strata-based weighting and prob-based hard sample boosting.
    
    Args:
        dataset: The dataset containing images with meta_list attribute
        target_pos_ratio: Desired ratio of positive samples (0.0 to 1.0)
        total_samples: Total number of samples to draw per epoch
        strata_weights: Optional dict like {'1_q4':1.0, '1_q3':0.8, ... '0_none':0.5}
                        Weight 0 means skip that strata. If None, default to equal weights.
    
    Returns:
        Either SubsetRandomSampler or WeightedRandomSampler
    """
    if strata_weights is None:
        strata_weights = {'1_q4': 1.0, '1_q3': 1.0, '1_q2': 1.0, '1_q1': 1.0, '0_none': 1.0}
    
    # Step 1: Group indices by strata and prepare hard_weights
    strata_indices = defaultdict(list)
    hard_weights = np.ones(len(dataset))  # Default weight 1.0 for each index
    for i in range(len(dataset)):
        meta = dataset.meta_list[i]
        if 'strata' not in meta:
            raise ValueError(f"Meta {i} missing 'strata'! Check preparing_data.py")
        strata = meta['strata']
        strata_indices[strata].append(i)
        
        # Boost hard samples based on prob (from classifier)
        if 'cls_prob' in meta:
            if not meta['has_pnx'] and meta['cls_prob'] > 0.5:  # hard negative
                hard_weights[i] = 2.0
            elif meta['has_pnx'] and meta['cls_prob'] < 0.5:  # hard positive
                hard_weights[i] = 1.5
        else:
            print(f"Warning: Meta {i} missing 'cls_prob', no hard boost.")  # Debug hint
    
    # Step 2: Identify pos/neg strata and compute summary weights
    pos_strata = [k for k in strata_weights if k.startswith('1_')]
    neg_strata = ['0_none'] if '0_none' in strata_weights else []
    sum_pos_w = sum(strata_weights.get(k, 0) for k in pos_strata)
    sum_neg_w = strata_weights.get('0_none', 0)
    
    if sum_pos_w == 0 and target_pos_ratio > 0:
        raise ValueError("No positive strata weights, but pos_ratio >0!")
    if sum_neg_w == 0 and target_pos_ratio < 1:
        raise ValueError("No negative strata weights, but need neg samples!")
    
    pos_samples = int(total_samples * target_pos_ratio)
    neg_samples = total_samples - pos_samples
    
    selected_indices = []
    
    # Step 3: Sample from positive strata
    for k in pos_strata:
        w = strata_weights.get(k, 0)
        if w > 0:
            if sum_pos_w > 0:
                alloc = int(pos_samples * (w / sum_pos_w))
            else:
                alloc = 0
            indices = strata_indices.get(k, [])
            if alloc > 0 and indices:
                local_weights = hard_weights[indices]
                if local_weights.sum() > 0:
                    local_weights = local_weights / local_weights.sum() 
                else:
                    np.ones(len(indices)) / len(indices)
                replace = (alloc > len(indices))
                chosen = np.random.choice(indices, alloc, replace=replace, p=local_weights)
                selected_indices.extend(chosen)
    
    # Step 4: Sample from negative strata (usually just '0_none')
    for k in neg_strata:
        w = strata_weights.get(k, 0)
        if w > 0:
            alloc = int(neg_samples * (w / sum_neg_w)) if sum_neg_w > 0 else 0
            indices = strata_indices.get(k, [])
            if alloc > 0 and indices:
                local_weights = hard_weights[indices]
                local_weights = local_weights / local_weights.sum() if local_weights.sum() > 0 else np.ones(len(indices)) / len(indices)
                replace = (alloc > len(indices))
                chosen = np.random.choice(indices, alloc, replace=replace, p=local_weights)
                selected_indices.extend(chosen)
    
    # Step 5: If selected < total (due to rounding or empty strata), fall back to weighted sampling
    if len(selected_indices) < total_samples:
        print(f"Warning: Selected {len(selected_indices)} < {total_samples}, using WeightedRandomSampler with replacement.")
        all_weights = np.zeros(len(dataset))
        for k, inds in strata_indices.items():
            strata_w = strata_weights.get(k, 0)
            all_weights[inds] = strata_w * hard_weights[inds]  # Combine strata and hard
        all_weights /= all_weights.sum() if all_weights.sum() > 0 else 1
        return WeightedRandomSampler(all_weights, total_samples, replacement=True)
    else:
        # Shuffle and trim if over (rare, but safe)
        np.random.shuffle(selected_indices)
        return SubsetRandomSampler(selected_indices[:total_samples])


"""origin sampler
def create_ratio_based_sampler(
    dataset, 
    target_pos_ratio, 
    total_samples,
):
    # Step 1: Separate indices based on pneumothorax presence
    positive_indices, negative_indices = [], []
    for i in range(len(dataset)):
        if dataset.meta_list[i]['has_pnx']:  # has pneumothorax
            positive_indices.append(i)
        else:  # no pneumothorax
            negative_indices.append(i)

    # Step 2: Calculate desired sample counts for each class
    pos_samples = int(total_samples * target_pos_ratio)     # e.g., 0.8 * 1000 = 800
    neg_samples = total_samples - pos_samples               # e.g., 1000 - 800 = 200

    # Step 3: Choose sampling strategy based on data availability
    if pos_samples <= len(positive_indices) and neg_samples <= len(negative_indices):
        # CASE 1: We have enough samples in both classes - use subset sampling
        # Randomly select exact number of samples from each class WITHOUT replacement
        selected_pos = np.random.choice(positive_indices, pos_samples, replace=False)
        selected_neg = np.random.choice(negative_indices, neg_samples, replace=False)
        
        # Combine and shuffle the selected indices
        selected_indices = np.concatenate([selected_pos, selected_neg])
        np.random.shuffle(selected_indices)
        
        return SubsetRandomSampler(selected_indices)
    else:
        # CASE 2: Not enough samples in one/both classes - use weighted sampling
        # This allows sampling WITH replacement to achieve desired ratios
        all_weights = np.zeros(len(dataset))
        
        # Calculate weights: higher weight = more likely to be sampled
        # If we need more pos samples than available, increase weight proportionally
        pos_weight = pos_samples / len(positive_indices) if pos_samples > len(positive_indices) else 1.0
        neg_weight = neg_samples / len(negative_indices) if neg_samples > len(negative_indices) else 1.0
        
        # Assign weights to corresponding indices
        all_weights[positive_indices] = pos_weight
        all_weights[negative_indices] = neg_weight
        
        return WeightedRandomSampler(all_weights, total_samples, replacement=True)
"""
MODEL_BUILDERS = {
    'deeplabv3plus': get_DeepLabV3Plus,
    'unet': get_Unet,
    'fpn': get_FPN_ResNet50,
    'pspnet': get_PSPNet_ResNet50,
}

def run_pipeline(stages, config, modelname):

    best_model_path = None
    time_records = []

    for idx, stage in enumerate(stages):
        print(f"\n=== Stage {idx} ===")
        print(f"lr={stage['lr']}, sample_rate={stage['sample_rate']}, image_size={stage['image_size']}")

        train_transform, val_transform = get_transform(image_size=stage['image_size'])

        train_dataset = pnxImgSegSet(
            datapath=config['root_path'], 
            meta_list_path=config['meta_path'], 
            mask_key=config['mask_key'],
            skip_has_pnx=config['skip_has_pnx'],
            split='train', 
            transform=train_transform, 
            image_size=stage['image_size'],
            calc_class_weights=config['calc_class_weights'],
        )
        print("Checking training dataset...")
        _ = validate_dataset(train_dataset)
        print(f'Training samples: {len(train_dataset)}')

        val_dataset = pnxImgSegSet(
            datapath=config['root_path'], 
            meta_list_path=config['meta_path'], 
            mask_key=config['mask_key'],
            split='test', 
            transform=val_transform, 
            image_size=stage['image_size'],
        )
        print("Checking validation dataset...")
        _ = validate_dataset(val_dataset)
        print(f'Validation samples: {len(val_dataset)}')

        # 動態建立 DataLoader
        sampler = create_ratio_based_sampler(
            dataset=train_dataset,
            target_pos_ratio=stage["sample_rate"],
            total_samples=len(train_dataset),
            strata_weights=stage.get("strata_weights")
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=config['bsz'],
            sampler=sampler,
            num_workers=4,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['bsz'],
            shuffle=False,
            num_workers=4,
        )
        
        criterion = get_lossFunc(
            lossFunc=config['criterion'],
            class_weights=train_dataset.weights
        )

        print("\n" + "="*60)
        print(f"Training {modelname}")
        print("="*60)
        model = MODEL_BUILDERS[modelname](
            device=config['device'],
        )
        
        # 載入前一階段的最佳權重
        if idx > 0 and best_model_path:
            if os.path.exists(best_model_path):
                checkpoint = torch.load(best_model_path, weights_only=True)
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"Loading best {modelname} weights from: {best_model_path}")
            else:
                raise FileNotFoundError(f"{best_model_path} does not exist!")
        
        num_epochs = stage['epochs']
        steps_per_epoch = len(train_loader)
        optimizer, scheduler = get_optim(
            model=model,
            lr=stage["lr"],
            scheduler_type=stage["scheduler"],
            num_epochs=num_epochs,
            steps_per_epoch=steps_per_epoch
        )
        
        start_time = time.time()

        history = train_model(
            model=model, 
            train_loader=train_loader, 
            val_loader=val_loader, 
            criterion=criterion, 
            optimizer=optimizer, 
            scheduler=scheduler, 
            num_epochs=stage['epochs'], 
            device=config['device'], 
            model_name=f'{modelname}_stage{idx}', 
            save_dir=config['save_path'],
        )

        elapsed = time.time() - start_time
        print(f"Stage {idx} training time: {elapsed:.2f} sec")

        # 儲存紀錄
        time_records.append({
            "model": modelname,
            "stage": idx,
            "epochs": stage['epochs'],
            "lr": stage['lr'],
            "elapsed_sec": elapsed
        })
        
        # 更新最佳模型路徑
        best_model_path = os.path.join(config['save_path'], f'best_{modelname}_stage{idx}.pth')

    times_file = os.path.join(config['save_path'], f"{modelname}_train_times.json")
    with open(times_file, "w") as f:
        json.dump(time_records, f, indent=4)
    print(f"Saved training times to {times_file}")

    return {f'best_{modelname}_path': best_model_path}

def main():
    config = {
        'bsz': 12,
        'mask_key': 'cropped_mask_path',
        'skip_has_pnx': False,
        'calc_class_weights': True,
        'criterion': 'improvedcombined',
        'root_path': os.getcwd(),
        'meta_path': 'subset_data_2508221615.json',
        'save_path': os.path.join(os.getcwd(), 'checkpoints', datetime.now().strftime("%y%m%d%H%M")),
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    }
    print(os.path.join(os.getcwd(), 'checkpoints', datetime.now().strftime("%y%m%d%H%M")))
    os.makedirs(config['save_path'], exist_ok=True)

    stages = [
    {'epochs': 30, 'lr': 3e-4, 'sample_rate': 0.7, 'image_size': (768, 768), 'scheduler': 'OneCycleLR',
         'strata_weights': {'1_q4': 0.8, '1_q3': 0.7, '1_q2': 0.7, '1_q1': 0.6, '0_none': 0.2}},
    {'epochs': 30, 'lr': 1e-4, 'sample_rate': 0.5, 'image_size': (768, 768), 'scheduler': 'OneCycleLR',
         'strata_weights': {'1_q4': 0.5, '1_q3': 0.5, '1_q2': 0.8, '1_q1': 0.8, '0_none': 0.4}},
    {'epochs': 40, 'lr': 1e-5, 'sample_rate': 0.2, 'image_size': (768, 768), 'scheduler': 'OneCycleLR',
         'strata_weights': {'1_q4': 0.2, '1_q3': 0.2, '1_q2': 1.0, '1_q1': 1.0, '0_none': 0.6}}
    ] 

    model_names = ['deeplabv3plus', 'unet', 'fpn', 'pspnet']
    results = {}
    for modelname in model_names:
        print(f"\n=== Training pipeline for {modelname} ===") 
        result = run_pipeline(stages, config, modelname)
        results.update(result)

    # 可選擇繪製比較圖（需要修改 plot_training_comparison 來支援多模型）
    # plot_training_comparison(results, save_dir=config['save_path'])

if __name__ == "__main__":
    main()