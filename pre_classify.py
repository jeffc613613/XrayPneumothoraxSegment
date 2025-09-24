import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt


class PneumothoraxDataset(Dataset):
    def __init__(self, data_list, transform=None):
        self.data_list = data_list
        self.transform = transform
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        item = self.data_list[idx]
        
        # Load image
        try:
            image = Image.open(item['image_path']).convert('RGB')
        except:
            print(f"Warning: Unable to load image {item['image_path']}")
            image = Image.new('RGB', (512, 512), (0, 0, 0))
        
        if self.transform:
            image = self.transform(image)
        
        # Float label for BCEWithLogitsLoss, shape [1]
        label = 1.0 if item['has_pnx'] else 0.0
        return image, torch.tensor(label, dtype=torch.float).unsqueeze(0)


class PneumothoraxClassifier(nn.Module):
    def __init__(self):
        super(PneumothoraxClassifier, self).__init__()
        # Use pretrained ResNet50
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        # Change last layer to output single logit for binary classification
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
        x = self.backbone.fc(x)
        return x  # raw logits shape: [batch_size, 1]


def load_data(json_path):
    """Load data from JSON"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    train_data = [item for item in data if item.get('split') == 'train']
    val_data = [item for item in data if item.get('split') == 'val']
    test_data = [item for item in data if item.get('split') == 'test']
    
    if train_data:
        # Split train_data into 80% train, 20% val
        np.random.shuffle(train_data)
        n_train = int(0.8 * len(train_data))
        val_data_new = train_data[n_train:]
        train_data_new = train_data[:n_train]
        
        if val_data:
            val_data = val_data + val_data_new
        else:
            val_data = val_data_new
        train_data = train_data_new
    
    elif not train_data and not val_data and not test_data:
        np.random.shuffle(data)
        n = len(data)
        train_data = data[:int(0.7*n)]
        val_data = data[int(0.7*n):int(0.85*n)]
        test_data = data[int(0.85*n):]
    
    print(f"Training set: {len(train_data)}")
    print(f"Validation set: {len(val_data)}")
    print(f"Test set: {len(test_data)}")
    return train_data, val_data, test_data


def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    
    return train_transform, val_transform


def train_model(model, train_loader, val_loader, num_epochs=50, lr=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
    
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    best_val_acc = 0.0
    best_model_state = None
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        train_iter = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} Training", leave=False)
        for images, labels in train_iter:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)  # shape [batch_size, 1]
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_iter.set_postfix(loss=loss.item())
        
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        val_iter = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} Validation", leave=False)
        with torch.no_grad():
            for images, labels in val_iter:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                probs = torch.sigmoid(outputs)
                predicted = (probs > 0.5).float()
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                val_iter.set_postfix(loss=loss.item())
        
        val_acc = 100 * correct / total
        
        train_losses.append(train_loss / len(train_loader))
        val_losses.append(val_loss / len(val_loader))
        val_accuracies.append(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
        
        scheduler.step()
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {train_loss/len(train_loader):.4f}')
        print(f'Val Loss: {val_loss/len(val_loader):.4f}')
        print(f'Val Accuracy: {val_acc:.2f}%')
        print('-' * 50)
    
    model.load_state_dict(best_model_state)
    return model, train_losses, val_losses, val_accuracies


def evaluate_model(model, test_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    y_true = []
    y_pred = []
    
    test_iter = tqdm(test_loader, desc="Evaluating", leave=False)
    with torch.no_grad():
        for images, labels in test_iter:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            predicted = (probs > 0.5).float()
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    
    print("Test set evaluation results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    
    return accuracy, precision, recall, f1, cm


def plot_training_curves(train_losses, val_losses, val_accuracies):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    ax2.plot(val_accuracies, label='Validation Accuracy')
    ax2.set_title('Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()


def main():
    JSON_PATH = "D:\XrayPnxSegment\subset_data_12000.json"
    BATCH_SIZE = 32
    NUM_EPOCHS = 15
    LEARNING_RATE = 0.001
    
    print("Loading data...")
    train_data, val_data, test_data = load_data(JSON_PATH)
    
    train_pos = sum(1 for item in train_data if item['has_pnx'])
    train_neg = len(train_data) - train_pos
    print(f"Training set - Positive: {train_pos}, Negative: {train_neg}")
    
    train_transform, val_transform = get_transforms()
    
    train_dataset = PneumothoraxDataset(train_data, train_transform)
    val_dataset = PneumothoraxDataset(val_data, val_transform)
    test_dataset = PneumothoraxDataset(test_data, val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = PneumothoraxClassifier()
    
    print("Starting training...")
    model, train_losses, val_losses, val_accuracies = train_model(
        model, train_loader, val_loader, num_epochs=NUM_EPOCHS, lr=LEARNING_RATE
    )
    
    plot_training_curves(train_losses, val_losses, val_accuracies)
    
    print("Evaluating the model...")
    accuracy, precision, recall, f1, cm = evaluate_model(model, test_loader)
    
    torch.save(model.state_dict(), 'res18_pneumothorax_classifier.pth')
    print("Model saved as pneumothorax_classifier.pth")


if __name__ == "__main__":
    main()