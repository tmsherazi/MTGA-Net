# MTGA-Net: Multi-Tier Global Attention Network for Breast Mass Classification

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)
from torchcam.methods import GradCAM
from torchcam.utils import overlay_mask
import torchvision.transforms.functional as F_tv


# ----------------------------
# Branch Attention Module (BAM)
# ----------------------------
class BranchAttentionModule(nn.Module):
    def __init__(self, reduction_ratio=16):
        super(BranchAttentionModule, self).__init__()
        C1, C2, C3 = 256, 512, 1024  # ResNet152 stage output channels

        self.dnn1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(C1, C1 // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(C1 // reduction_ratio, C2),
            nn.Softmax(dim=1)
        )

        self.dnn2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(C2, C2 // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(C2 // reduction_ratio, C3),
            nn.Softmax(dim=1)
        )

        self.dnn3 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(C3, C3 // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(C3 // reduction_ratio, 2048),
            nn.Softmax(dim=1)
        )

    def forward(self, x1, x2, x3, x4):
        b1 = self.dnn1(x1).view(x1.size(0), -1, 1, 1)
        x2_att = x2 * b1

        b2 = self.dnn2(x2_att).view(x2_att.size(0), -1, 1, 1)
        x3_att = x3 * b2

        b3 = self.dnn3(x3_att).view(x3_att.size(0), -1, 1, 1)
        M = x4 * b3
        return M


# ----------------------------
# MTGA-Net Model
# ----------------------------
class MTGANet(nn.Module):
    def __init__(self, num_classes=2):
        super(MTGANet, self).__init__()
        resnet = models.resnet152(pretrained=True)

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1  # 256
        self.layer2 = resnet.layer2  # 512
        self.layer3 = resnet.layer3  # 1024
        self.layer4 = resnet.layer4  # 2048

        self.bam = BranchAttentionModule()
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

        M = self.bam(x1, x2, x3, x4)
        out = self.avgpool(M)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


# ----------------------------
# Custom Dataset with Augmentation
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
            angle = torch.randint(-15, 16, (1,)).item()
            image = F_tv.rotate(image, angle, fill=(0,))

        if self.transform:
            image = self.transform(image)

        return image, label


# ----------------------------
# Data Loading Function (Your Exact Paths)
# ----------------------------
def load_data(base_dir, split='train'):
    """
    Load data from your specified Google Drive structure.
    Note: 'maligant' spelling is preserved as per your input.
    """
    if split == 'train':
        benign_dir = os.path.join(base_dir, "Train", "benign")
        malignant_dir = os.path.join(base_dir, "Train", "maligant")
    elif split == 'val':
        benign_dir = os.path.join(base_dir, "Validation", "benign")
        malignant_dir = os.path.join(base_dir, "Validation", "maligant")
    else:
        raise ValueError("split must be 'train' or 'val'")

    paths, labels = [], []

    # Benign
    if os.path.exists(benign_dir):
        for fname in os.listdir(benign_dir):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                paths.append(os.path.join(benign_dir, fname))
                labels.append(0)
    else:
        print(f"Warning: benign directory not found: {benign_dir}")

    # Malignant (note: uses 'maligant' as per your path)
    if os.path.exists(malignant_dir):
        for fname in os.listdir(malignant_dir):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                paths.append(os.path.join(malignant_dir, fname))
                labels.append(1)
    else:
        print(f"Warning: malignant directory not found: {malignant_dir}")

    return paths, labels


# ----------------------------
# Training Function
# ----------------------------
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=40, device='cuda'):
    model.to(device)
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_auc = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Validation
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        y_true_val, y_scores_val = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                y_true_val.extend(labels.cpu().numpy())
                y_scores_val.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().numpy())

        val_loss /= len(val_loader)
        val_acc = 100 * correct / total
        val_auc = roc_auc_score(y_true_val, y_scores_val)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        scheduler.step()

        print(f'Epoch {epoch + 1:02d}/{num_epochs} | '
              f'Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | '
              f'Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%, AUC: {val_auc:.5f}')

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), 'mtga_net_best.pth')

    return train_losses, val_losses, train_accs, val_accs, best_auc


# ----------------------------
# Evaluation & Grad-CAM Visualization
# ----------------------------
def evaluate_and_visualize(model, test_loader, class_names, device='cuda', num_vis=5):
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

    auc_score = roc_auc_score(y_true, y_scores)
    print(f"\nTest AUC: {auc_score:.5f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.5f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()

    # Grad-CAM
    cam_extractor = GradCAM(model, target_layer='layer4')
    fig, axes = plt.subplots(2, num_vis, figsize=(3 * num_vis, 6))
    count = 0
    with torch.no_grad():
        for images, labels in test_loader:
            if count >= num_vis:
                break
            for i in range(images.size(0)):
                if count >= num_vis:
                    break
                img_tensor = images[i:i + 1].to(device)
                label = labels[i].item()
                output = model(img_tensor)
                pred = output.argmax(dim=1).item()
                confidence = torch.softmax(output, dim=1)[0, pred].item()

                activation_map = cam_extractor(pred, output)
                result = overlay_mask(
                    transforms.ToPILImage()(images[i]),
                    transforms.ToPILImage()(activation_map[0].cpu()),
                    alpha=0.6
                )

                row = 0 if pred == label else 1
                axes[row, count].imshow(result)
                axes[row, count].set_title(f'True: {class_names[label]}\nPred: {class_names[pred]} ({confidence:.2f})')
                axes[row, count].axis('off')
                count += 1

    plt.suptitle("Grad-CAM: Correct (Top) vs Incorrect (Bottom) Predictions")
    plt.tight_layout()
    plt.show()


# ----------------------------
# Main Execution
# ----------------------------
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ✅ YOUR EXACT PATH
    base_dir = "/content/drive/MyDrive/data"

    # Transforms
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

    # Load data using your structure
    train_paths, train_labels = load_data(base_dir, 'train')
    val_paths, val_labels = load_data(base_dir, 'val')

    print(f"Training samples: {len(train_paths)} (Benign: {train_labels.count(0)}, Malignant: {train_labels.count(1)})")
    print(f"Validation samples: {len(val_paths)} (Benign: {val_labels.count(0)}, Malignant: {val_labels.count(1)})")

    train_dataset = MammographyDataset(train_paths, train_labels, transform=train_transform, augment=True)
    val_dataset = MammographyDataset(val_paths, val_labels, transform=val_transform, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2, pin_memory=True)

    # Model
    model = MTGANet(num_classes=2)
    model.to(device)

    # Optimizer & Scheduler (as per your setup)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # Train
    print("\nStarting training (40 epochs, batch_size=8, SGD with LR decay)...\n")
    train_losses, val_losses, train_accs, val_accs, best_auc = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        num_epochs=40, device=device
    )

    # Plot curves
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch');
    plt.ylabel('Loss');
    plt.legend();
    plt.title('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.xlabel('Epoch');
    plt.ylabel('Accuracy (%)');
    plt.legend();
    plt.title('Accuracy')
    plt.tight_layout()
    plt.show()

    # Final evaluation
    print("\nEvaluating best model (by AUC)...")
    model.load_state_dict(torch.load('mtga_net_best.pth'))
    evaluate_and_visualize(model, val_loader, ['Benign', 'Malignant'], device=device)

    print(f"\n✅ Training completed. Best validation AUC: {best_auc:.5f}")


if __name__ == "__main__":
    main()