import torch
from XrayPnxSegment.common.utils import predict_and_visualize
from XrayPnxSegment.processors.img_processor import get_transform
from XrayPnxSegment.models.modeling_segModels import get_DeepLabV3Plus
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = get_DeepLabV3Plus()
model.load_state_dict(torch.load(r'D:\XrayPnxSegment\checkpoints\DeepLabV3\best_deeplabv3plus_stage2.pth')['model_state_dict'])

predict_and_visualize(
    model=model, 
    image_path=r'D:\XrayPnxSegment\siim-acr-pneumothorax\png_images\74_test_1_.png', 
    mask_path=r'D:\XrayPnxSegment\siim-acr-pneumothorax\png_masks\74_test_1_.png', 
    device=device,
    transform=get_transform()[1], 
)