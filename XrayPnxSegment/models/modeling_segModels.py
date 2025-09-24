"""
 SPDX-License-Identifier: MIT
 Copyright (c) 2025, yasaisen (clover)
 
 This file is part of a project licensed under the MIT License.
 See the LICENSE file in the project root for more information.
 
 last modified in 2506222348
"""

import torch
import segmentation_models_pytorch as smp
import torchvision.models.segmentation as seg_models

from ..common.utils import get_trainable_params


IMG_ENCODER = "resnet50"
IMG_ENCODER_WEIGHT = "imagenet"

def get_DeepLabV3Plus(
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    deeplabv3_model = smp.DeepLabV3Plus(
        encoder_name=IMG_ENCODER,
        encoder_weights=IMG_ENCODER_WEIGHT,
        in_channels=3,
        classes=1,
        activation=None
    ).to(device)
    print(get_trainable_params(deeplabv3_model))
    return deeplabv3_model

def get_Unet(
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    unet_model = smp.Unet(
        encoder_name=IMG_ENCODER,
        encoder_weights=IMG_ENCODER_WEIGHT,
        in_channels=3,
        classes=1,
        activation=None
    ).to(device)
    print(get_trainable_params(unet_model))
    return unet_model

def get_FPN_ResNet50(
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    fpn_model = smp.FPN(
        encoder_name=IMG_ENCODER,
        encoder_weights=IMG_ENCODER_WEIGHT,
        in_channels=3,
        classes=1,
        activation=None
    ).to(device)
    
    print(get_trainable_params(fpn_model))
    return fpn_model

def get_PSPNet_ResNet50(
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    psp_model = smp.PSPNet(
        encoder_name=IMG_ENCODER, 
        encoder_weights=IMG_ENCODER_WEIGHT,
        in_channels=3,
        classes=1,
        activation=None
    ).to(device)
    
    print(get_trainable_params(psp_model))
    return psp_model

class LRASPPWrapper(torch.nn.Module):
    def __init__(self, num_classes=1, pretrained=False, device="cuda"):
        super().__init__()
        if pretrained:
            self.model = seg_models.lraspp_mobilenet_v3_large(
                weights=seg_models.LRASPP_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1,
                num_classes=num_classes
            )
        else:
            self.model = seg_models.lraspp_mobilenet_v3_large(
                weights=None,
                num_classes=num_classes
            )
        self.model = self.model.to(device)
        self.encoder = self.model.backbone
        self.decoder = self.model.classifier
    
    def forward(self, x):
        outputs = self.model(x)
    
        if isinstance(outputs, dict):
            return outputs.get('out', list(outputs.values())[0])
        return outputs

def get_LRASPP_MobileNetV3(
    pretrained=False,
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    lraspp_model = LRASPPWrapper(
        num_classes=1,
        pretrained=pretrained,
        device=device
    )
    
    print(get_trainable_params(lraspp_model))
    return lraspp_model