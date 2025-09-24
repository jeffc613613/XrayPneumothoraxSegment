"""
 SPDX-License-Identifier: MIT
 Copyright (c) 2025, yasaisen (clover)
 
 This file is part of a project licensed under the MIT License.
 See the LICENSE file in the project root for more information.
 
 last modified in 2506222348
"""

import numpy as np
import json
import os
import matplotlib.pyplot as plt
import torch
import cv2

def load_json_data(
    file_path: str,
):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            return data
    except FileNotFoundError:
        print(f"Error: Data file not found {file_path}")
        return []
    except json.JSONDecodeError as e:
        print(f"Error: JSON parsing failed - {e}")
        return []
    
def convert(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (np.float32, np.float64, np.float16)):
        return float(obj)
    raise TypeError(f"Type {type(obj)} not serializable")

def get_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def plot_training_comparison(
    # deeplabv3_history, 
    # unet_history, 
    FPN_history,
    PSPNET_history,
    # LRASPP_history,
    save_dir,
):
    # fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # epochs = range(1, len(deeplabv3_history['train_loss']) + 1)
    
    # # Loss
    # axes[0, 0].plot(epochs, deeplabv3_history['train_loss'], 'b-', label='DeepLabV3Plus Train')
    # axes[0, 0].plot(epochs, deeplabv3_history['val_loss'], 'b--', label='DeepLabV3Plus Val')
    # axes[0, 0].plot(epochs, unet_history['train_loss'], 'r-', label='U-Net Train')
    # axes[0, 0].plot(epochs, unet_history['val_loss'], 'r--', label='U-Net Val')
    # axes[0, 0].set_title('Training and Validation Loss')
    # axes[0, 0].set_xlabel('Epochs')
    # axes[0, 0].set_ylabel('Loss')
    # axes[0, 0].legend()
    # axes[0, 0].grid(True)
    
    # # IoU
    # axes[0, 1].plot(epochs, deeplabv3_history['train_iou'], 'b-', label='DeepLabV3Plus Train')
    # axes[0, 1].plot(epochs, deeplabv3_history['val_iou'], 'b--', label='DeepLabV3Plus Val')
    # axes[0, 1].plot(epochs, unet_history['train_iou'], 'r-', label='U-Net Train')
    # axes[0, 1].plot(epochs, unet_history['val_iou'], 'r--', label='U-Net Val')
    # axes[0, 1].set_title('IoU Score')
    # axes[0, 1].set_xlabel('Epochs')
    # axes[0, 1].set_ylabel('IoU')
    # axes[0, 1].legend()
    # axes[0, 1].grid(True)
    
    # # Dice
    # axes[1, 0].plot(epochs, deeplabv3_history['train_dice'], 'b-', label='DeepLabV3Plus Train')
    # axes[1, 0].plot(epochs, deeplabv3_history['val_dice'], 'b--', label='DeepLabV3Plus Val')
    # axes[1, 0].plot(epochs, unet_history['train_dice'], 'r-', label='U-Net Train')
    # axes[1, 0].plot(epochs, unet_history['val_dice'], 'r--', label='U-Net Val')
    # axes[1, 0].set_title('Dice Coefficient')
    # axes[1, 0].set_xlabel('Epochs')
    # axes[1, 0].set_ylabel('Dice')
    # axes[1, 0].legend()
    # axes[1, 0].grid(True)
    
    # # Accuracy
    # axes[1, 1].plot(epochs, deeplabv3_history['train_accuracy'], 'b-', label='DeepLabV3Plus Train')
    # axes[1, 1].plot(epochs, deeplabv3_history['val_accuracy'], 'b--', label='DeepLabV3Plus Val')
    # axes[1, 1].plot(epochs, unet_history['train_accuracy'], 'r-', label='U-Net Train')
    # axes[1, 1].plot(epochs, unet_history['val_accuracy'], 'r--', label='U-Net Val')
    # axes[1, 1].set_title('Pixel Accuracy')
    # axes[1, 1].set_xlabel('Epochs')
    # axes[1, 1].set_ylabel('Accuracy')
    # axes[1, 1].legend()
    # axes[1, 1].grid(True)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs = range(1, len(FPN_history['train_loss']) + 1)
    
    # Loss
    axes[0, 0].plot(epochs, FPN_history['train_loss'], 'b-', label='FPN Train')
    axes[0, 0].plot(epochs, FPN_history['val_loss'], 'b--', label='FPN Val')
    # axes[0, 0].plot(epochs, PSPNET_history['train_loss'], 'r-', label='PSPNET Train')
    # axes[0, 0].plot(epochs, PSPNET_history['val_loss'], 'r--', label='PSPNET Val')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].set_xlabel('Epochs')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # IoU
    axes[0, 1].plot(epochs, FPN_history['train_iou'], 'b-', label='FPN Train')
    axes[0, 1].plot(epochs, FPN_history['val_iou'], 'b--', label='FPN Val')
    # axes[0, 1].plot(epochs, PSPNET_history['train_iou'], 'r-', label='PSPNET Train')
    # axes[0, 1].plot(epochs, PSPNET_history['val_iou'], 'r--', label='PSPNET Val')
    axes[0, 1].set_title('IoU Score')
    axes[0, 1].set_xlabel('Epochs')
    axes[0, 1].set_ylabel('IoU')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Dice
    axes[1, 0].plot(epochs, FPN_history['train_dice'], 'b-', label='FPN Train')
    axes[1, 0].plot(epochs, FPN_history['val_dice'], 'b--', label='FPN Val')
    # axes[1, 0].plot(epochs, PSPNET_history['train_dice'], 'r-', label='PSPNET Train')
    # axes[1, 0].plot(epochs, PSPNET_history['val_dice'], 'r--', label='PSPNET Val')
    axes[1, 0].set_title('Dice Coefficient')
    axes[1, 0].set_xlabel('Epochs')
    axes[1, 0].set_ylabel('Dice')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Accuracy
    axes[1, 1].plot(epochs, FPN_history['train_accuracy'], 'b-', label='FPN Train')
    axes[1, 1].plot(epochs, FPN_history['val_accuracy'], 'b--', label='FPN Val')
    # axes[1, 1].plot(epochs, PSPNET_history['train_accuracy'], 'r-', label='PSPNET Train')
    # axes[1, 1].plot(epochs, PSPNET_history['val_accuracy'], 'r--', label='PSPNET Val')
    axes[1, 1].set_title('Pixel Accuracy')
    axes[1, 1].set_xlabel('Epochs')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    # fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # epochs = range(1, len(LRASPP_history['train_loss']) + 1)
    
    # # Loss
    # axes[0, 0].plot(epochs, LRASPP_history['train_loss'], 'b-', label='LRASPP Train')
    # axes[0, 0].plot(epochs, LRASPP_history['val_loss'], 'b--', label='LRASPP Val')
    # # axes[0, 0].plot(epochs, PSPNET_history['train_loss'], 'r-', label='PSPNET Train')
    # # axes[0, 0].plot(epochs, PSPNET_history['val_loss'], 'r--', label='PSPNET Val')
    # axes[0, 0].set_title('Training and Validation Loss')
    # axes[0, 0].set_xlabel('Epochs')
    # axes[0, 0].set_ylabel('Loss')
    # axes[0, 0].legend()
    # axes[0, 0].grid(True)
    
    # # IoU
    # axes[0, 1].plot(epochs, LRASPP_history['train_iou'], 'b-', label='LRASPP Train')
    # axes[0, 1].plot(epochs, LRASPP_history['val_iou'], 'b--', label='LRASPP Val')
    # # axes[0, 1].plot(epochs, PSPNET_history['train_iou'], 'r-', label='PSPNET Train')
    # # axes[0, 1].plot(epochs, PSPNET_history['val_iou'], 'r--', label='PSPNET Val')
    # axes[0, 1].set_title('IoU Score')
    # axes[0, 1].set_xlabel('Epochs')
    # axes[0, 1].set_ylabel('IoU')
    # axes[0, 1].legend()
    # axes[0, 1].grid(True)
    
    # # Dice
    # axes[1, 0].plot(epochs, LRASPP_history['train_dice'], 'b-', label='LRASPP Train')
    # axes[1, 0].plot(epochs, LRASPP_history['val_dice'], 'b--', label='LRASPP Val')
    # # axes[1, 0].plot(epochs, PSPNET_history['train_dice'], 'r-', label='PSPNET Train')
    # # axes[1, 0].plot(epochs, PSPNET_history['val_dice'], 'r--', label='PSPNET Val')
    # axes[1, 0].set_title('Dice Coefficient')
    # axes[1, 0].set_xlabel('Epochs')
    # axes[1, 0].set_ylabel('Dice')
    # axes[1, 0].legend()
    # axes[1, 0].grid(True)
    
    # # Accuracy
    # axes[1, 1].plot(epochs, LRASPP_history['train_accuracy'], 'b-', label='LRASPP Train')
    # axes[1, 1].plot(epochs, LRASPP_history['val_accuracy'], 'b--', label='LRASPP Val')
    # # axes[1, 1].plot(epochs, PSPNET_history['train_accuracy'], 'r-', label='PSPNET Train')
    # # axes[1, 1].plot(epochs, PSPNET_history['val_accuracy'], 'r--', label='PSPNET Val')
    # axes[1, 1].set_title('Pixel Accuracy')
    # axes[1, 1].set_xlabel('Epochs')
    # axes[1, 1].set_ylabel('Accuracy')
    # axes[1, 1].legend()
    # axes[1, 1].grid(True)
    
    plt.tight_layout()
    savefig_path = os.path.join(save_dir, 'training_comparison.png')
    plt.savefig(savefig_path, dpi=300, bbox_inches='tight')
    plt.show()

def predict_and_visualize(
    model, 
    image_path, 
    mask_path,
    device, 
    transform=None
):
    model.eval()
    
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    image = cv2.resize(image, (768, 768), interpolation=cv2.INTER_NEAREST)
    mask = cv2.resize(mask, (768, 768), interpolation=cv2.INTER_NEAREST)
    
    if transform:
        augmented = transform(image=image)
        image_tensor = augmented['image']
    else:
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
    
    image_tensor = image_tensor.unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(image_tensor)
        if output.dim() == 4 and output.shape[1] == 1:
            output = output.squeeze(1)
        
        pred_mask = torch.sigmoid(output) > 0.5
        pred_mask = pred_mask.squeeze().cpu().numpy()
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(image)
    axes[0].set_title('image')
    
    axes[1].imshow(pred_mask, cmap='gray')
    axes[1].set_title('predicted Mask')
    
    axes[2].imshow(mask, cmap='gray')
    axes[2].set_title('groundtruth')
    
    plt.tight_layout()
    plt.show()
    














