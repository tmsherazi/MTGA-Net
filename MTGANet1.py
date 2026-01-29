# MTGA-Net: Multi-Tier Global-Attention Network

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)
from torchcam.methods import GradCAM
from torchcam.utils import overlay_mask
import torchvision.transforms.functional as F_tv


# ----------------------------
#
# ----------------------------
class BranchAttentionModule(nn.Module):
    def __init__(self, reduction_ratio=16):
        super(BranchAttentionModule, self).__init__()
        # ResNet152 stage output channels
        C1, C2, C3 = 256, 512, 1024

        # DNN1: stage1 → attention for stage2
        self.dnn1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(C1, C1 // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(C1 // reduction_ratio, C2),
            nn.Softmax(dim=1)
        )

        # DNN2: stage2 + b1 → attention for stage3
        self.dnn2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(C2, C2 // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(C2 // reduction_ratio, C3),
            nn.Softmax(dim=1)
        )

        # DNN3: stage3 + b2 → attention for stage4
        self.dnn3 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(C3, C3 // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(C3 // reduction_ratio, 2048),
            nn.Softmax(dim=1)
        )

    def forward(self, x1, x2, x3, x4):
        # Eq. (4): b1 = DNN1(g1)
        b1 = self.dnn1(x1).unsqueeze(-1).unsqueeze(-1)  # (B, 512, 1, 1)
        x2_weighted = x2 * b1

        # Eq. (6): b2 = DNN2(g2 · b1)
        b2 = self.dnn2(x2_weighted).unsqueeze(-1).unsqueeze(-1)  # (B, 1024, 1, 1)
        x3_weighted = x3 * b2

        # Eq. (7): b3 = DNN3(g3 · b2)
        b3 = self.dnn3(x3_weighted).unsqueeze(-1).unsqueeze(-1)  # (B, 2048, 1, 1)

        # Eq. (8): M = X4 · b3
        M = x4 * b3
        return M


# ----------------------------
# MTGA-Net Model (ResNet152V3 + BAM)
# ----------------------------
class MTGANet(nn.Module):
    def __init__(self, num_classes=2):
        super(MTGANet, self).__init__()
        resnet = models.resnet152(pretrained=True)

        # Backbone layers (frozen BN optional)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1  # 256, 56x56
        self.layer2 = resnet.layer2  # 512, 28x28
        self.layer3 = resnet.layer3  # 1024, 14x14
        self.layer4 = resnet.layer4  # 2048, 7x7

        # Proposed BAM
        self.bam = BranchAttentionModule()

        # Classifier head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        M = self.bam(x1, x2, x3, x4)  # Multi-tier refined features

        out = self.avgpool(M)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


# ----------------------------
# Dataset Class (with Augmentation)
# ----------------------------
class MammographyDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None, augment=False):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.augment = augment

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.augment:
            if torch.rand(1).item() < 0.5:
                image = F_tv.hflip(image)
            if torch.rand(1).item() < 0.5:
                image = F_tv.vflip(image)
            angle = int(torch.randint(-15, 16, (1,)).item())
            image = F_tv.rotate(image, angle, fill=(0,))

        if self.transform:
            image = self.transform(image)

        return image, label


# ----------------------------
# Data Loading (Exact Paths from Paper Setup)
# ----------------------------
def load_data(base_dir, split='train'):
    if split == 'train':
        benign_dir = os.path.join(base_dir, "Train", "benign")
        malignant_dir = os.path.join(base_dir, "Train", "maligant")  # as per your path
    elif split == 'val':
        benign_dir = os.path.join(base_dir, "Validation", "benign")
        malignant_dir = os.path.join(base_dir, "Validation", "maligant")
    else:
        raise ValueError("split must be 'train' or 'val'")

    paths, labels = [], []

    for fname in os.listdir(benign_dir):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            paths.append(os.path.join(benign_dir, fname))
            labels.append(0)

    for fname in os.listdir(malignant_dir):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            paths.append(os.path.join(malignant_dir, fname))
            labels.append(1)

    return paths, labels


# ----------------------------
# Training Function
# ----------------------------
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=40, device='cuda'):
    model.to(device)
    best_auc = 0.0

    for epoch in range(num_epochs):
        model.train()
        correct, total = 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = 100 * correct / total

        # Validation
        model.eval()
        y_true, y_scores = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)[:, 1]
                y_true.extend(labels.cpu().numpy())
                y_scores.extend(probs.cpu().numpy())

        val_auc = roc_auc_score(y_true, y_scores)
        scheduler.step()

        print(f'Epoch {epoch + 1:02d}/{num_epochs} | Train Acc: {train_acc:.2f}% | Val AUC: {val_auc:.5f}')

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), 'mtga_net_best.pth')

    return best_auc


# ----------------------------
# Evaluation & Grad-CAM
# ----------------------------
def evaluate_and_visualize(model, test_loader, device='cuda'):
    model.load_state_dict(torch.load('mtga_net_best.pth'))
    model.eval()
    y_true, y_pred, y_scores = [], [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_scores.extend(probs[:, 1].cpu().numpy())

    auc = roc_auc_score(y_true, y_scores)
    print(f"\n Final Test AUC: {auc:.5f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['Benign', 'Malignant']))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Benign', 'Malignant'], yticklabels=['Benign', 'Malignant'])
    plt.title('Confusion Matrix');
    plt.show()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.5f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate');
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve');
    plt.legend();
    plt.show()

    # Grad-CAM
    cam_extractor = GradCAM(model, target_layer='layer4')
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    count = 0
    with torch.no_grad():
        for images, labels in test_loader:
            if count >= 8: break
            for i in range(images.size(0)):
                if count >= 8: break
                img_tensor = images[i:i + 1].to(device)
                label = labels[i].item()
                output = model(img_tensor)
                pred = output.argmax().item()
                confidence = torch.softmax(output, dim=1)[0, pred].item()

                activation_map = cam_extractor(pred, output)
                result = overlay_mask(
                    transforms.ToPILImage()(images[i]),
                    transforms.ToPILImage()(activation_map[0].cpu()),
                    alpha=0.6
                )

                row = 0 if pred == label else 1
                col = count % 4
                axes[row, col].imshow(result)
                axes[row, col].set_title(
                    f'True: {"Benign" if label == 0 else "Malignant"}\nPred: {"Benign" if pred == 0 else "Malignant"} ({confidence:.2f})')
                axes[row, col].axis('off')
                count += 1

    plt.suptitle("Grad-CAM: Correct (Top) vs Incorrect (Bottom) Predictions")
    plt.tight_layout();
    plt.show()


# ----------------------------
# Main Execution
# ----------------------------
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ✅ EXACT PATH FROM YOUR SETUP
    base_dir = "/content/drive/MyDrive/data"

    # Transforms (ImageNet normalization)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])

    # Load data
    train_paths, train_labels = load_data(base_dir, 'train')
    val_paths, val_labels = load_data(base_dir, 'val')

    print(f"Train: {len(train_paths)} samples | Val: {len(val_paths)} samples")

    # Datasets & Loaders
    train_dataset = MammographyDataset(train_paths, train_labels, transform=train_transform, augment=True)
    val_dataset = MammographyDataset(val_paths, val_labels, transform=val_transform, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2, pin_memory=True)

    # Model
    model = MTGANet(num_classes=2)

    # Optimizer & Scheduler (Section IV.A)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # Train
    print("\n Starting training (40 epochs, batch=8, SGD, LR decay every 5 epochs)...\n")
    best_auc = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=40,
                           device=device)

    # Evaluate
    print("\n Evaluating best model...")
    evaluate_and_visualize(model, val_loader, device=device)

    print(f"\n Training complete. Best validation AUC: {best_auc:.5f}")


if __name__ == "__main__":
    main()