import torch
import torch.nn as nn
import cv2
import numpy as np
from torchvision import models

class PneumothoraxClassifier(nn.Module):
    """
    Pneumothorax binary classification model using ResNet18 backbone
    """
    def __init__(self):
        super(PneumothoraxClassifier, self).__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.backbone.fc(x)  # raw logits

class PneumothoraxPredictor:
    """
    Pneumothorax probability predictor wrapper using OpenCV for preprocessing
    """
    def __init__(self, model_path, device=None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = PneumothoraxClassifier()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.to(self.device).eval()
        # normalization constants
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        # image: BGR uint8 from cv2.imread
        # convert to RGB
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # resize
        img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
        # to float32 and normalize to [0,1]
        img = img.astype(np.float32) / 255.0
        # normalize per channel
        img = (img - self.mean) / self.std
        # HWC to CHW
        img = np.transpose(img, (2, 0, 1))
        # to torch tensor
        tensor = torch.from_numpy(img).unsqueeze(0).to(self.device)
        return tensor

    def predict_probability(self, image_path: str) -> float:
        """
        Predict probability from file path using cv2.imread
        """
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: cannot load image {image_path}")
            return None
        tensor = self._preprocess(img)
        with torch.no_grad():
            logits = self.model(tensor)
            return torch.sigmoid(logits).item()

    def predict_from_array(self, image_array: np.ndarray) -> float:
        """
        Predict probability from a BGR or RGB numpy array.
        If array is RGB, convert to BGR first.
        """
        img = image_array
        # if it's RGB, convert to BGR
        if img.shape[2] == 3 and img.dtype == np.uint8:
            # Heuristic: if max >1 and values look like RGB, convert
            # But assume input is BGR for cv2
            pass
        tensor = self._preprocess(img)
        with torch.no_grad():
            logits = self.model(tensor)
            return torch.sigmoid(logits).item()

    def predict_batch_probabilities(self, image_paths: list) -> list:
        """
        Batch predict from list of file paths
        """
        return [self.predict_probability(p) for p in image_paths]

def predict_pneumothorax(model_path: str, image_path: str, device=None) -> float:
    """
    Quick function: path in, probability out
    """
    predictor = PneumothoraxPredictor(model_path, device)
    return predictor.predict_probability(image_path)

# if __name__ == "__main__":
#     model_path = "res18_pneumothorax_classifier.pth"
#     predictor = PneumothoraxPredictor(model_path)
    # prob = predictor.predict_probability("chest_xray.jpg")
    # print(f"Pneumothorax probability: {prob:.4f}")

#     # Batch example
#     imgs = ["img1.jpg", "img2.jpg", "img3.jpg"]
#     probs = predictor.predict_batch_probabilities(imgs)
#     for i, p in enumerate(probs, 1):
#         print(f"Image {i} probability: {p:.4f}")