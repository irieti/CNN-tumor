from fnmatch import translate
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision import models

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from tqdm import tqdm

# configuration and hyperparameters
BATCH_SIZE = 32
NUM_EPOCHS = 25
LEARNING_RATE = 0.001
IMG_SIZE = 224
NUM_CLASSES = 4

device = torch.device("cpu")

TRAIN_DIR = "/archive/Training"
TEST_DIR = "/archive/Testing"


# dataset class
class BrainTumorDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.images = []
        self.labels = []

        self.class_to_idx = {"glioma": 0, "meningioma": 1, "notumor": 2, "pituitary": 3}

        print("Scanning for images in class folders ...")


device = torch.device("cpu")

TRAIN_DIR = "/Users/irie/Downloads/archive/Training"
TEST_DIR = "/Users/irie/Downloads/archive/Testing"


# dataset class
class BrainTumorDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.images = []
        self.labels = []

        self.class_to_idx = {"glioma": 0, "meningioma": 1, "notumor": 2, "pituitary": 3}

        print("Scanning for images in class folders...")

        for class_name, class_idx in self.class_to_idx.items():
            class_dir = os.path.join(data_dir, class_name)

            if not os.path.exists(class_dir):
                print(f"Warning! {class_dir} does not exist")
                continue

            class_images = 0

            for img_name in os.listdir(class_dir):
                if img_name.endswith((".jpg", ".jpeg", ".png")):
                    img_path = os.path.join(class_dir, img_name)
                    self.images.append(img_path)
                    self.labels.append(class_idx)
                    class_images += 1

        print(f"Total images loaded: {len(self.images)}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        img_path = self.images[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


# Data transformation

train_transforms = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Validation/Test transfromation

val_transforms = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

if __name__ == "__main__":

    train_dataset = BrainTumorDataset(data_dir=TRAIN_DIR, transform=train_transforms)

    test_dataset = BrainTumorDataset(data_dir=TEST_DIR, transform=val_transforms)

    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size

    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    print(f"Training samples: {train_size}")
    print(f"Validation samples: {val_size}")

    num_workers = 0
    pin_memory = False

    train_loader = DataLoader(
        train_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    print(f"Training batches per epoch: {len(train_loader)}")

    val_loader = DataLoader(
        val_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    print(f"Validation bayches per epoch: {len(val_loader)}")

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    print(f"Test batches: {len(test_loader)}")

    print(f"Training samples: {len(train_subset)}")
    print(f"Validation samples: {len(val_subset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Number of classes: {NUM_CLASSES}")

    # CNN model architecture

    class BrainTumorCNN(nn.Module):

        def __init__(self, num_classes=4):

            super(BrainTumorCNN, self).__init__()

            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Dropout(0.25),
            )

            self.conv2 = nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                nn.Dropout(0.25),
            )

            self.conv3 = nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                nn.Dropout(0.25),
            )

            self.conv4 = nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                nn.Dropout(0.25),
            )

            self.fc1 = nn.Linear(256 * 14 * 14, 512)
            self.dropout_fc = nn.Dropout(0.5)
            self.fc2 = nn.Linear(512, num_classes)

        def forward(self, x):

            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.conv4(x)

            x = x.view(x.size(0), -1)

            x = torch.relu(self.fc1(x))
            x = self.dropout_fc(x)

            x = self.fc2(x)

            return x


# model, loss and optimizers init

model = BrainTumorCNN(num_classes=NUM_CLASSES).to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

sheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=3
)

# training function


def train_epoch(model, dataloader, criterion, optimizer, device):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    batch_num = 0
    for images, labels in tqdm(dataloader, desc="Training", leave=False):

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item() * images.size(0)

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        batch_num += 1
        if batch_num % 10 == 0:
            batch_acc = 100 * correct / total
            batch_loss = running_loss / total
            print(
                f"Batch {batch_num}/{len(dataloader)}, Loss: {batch_loss:.4f}, Acc: {batch_acc:.2f}%"
            )

    epoch_loss = running_loss / total
    epoch_acc = 100 * correct / total

    return epoch_loss, epoch_acc


# validation


def validate(model, dataloader, criterion, device):

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validating", leave=False):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100 * correct / total

    print(f" Validation complete - {correct}/{total} samples correct")

    return epoch_loss, epoch_acc


# training
training_losses = []
train_accs = []
val_losses = []
val_accs = []

best_val_acc = 0.0

print("Starting Training")

for epoch in range(NUM_EPOCHS):

    current_lr = optimizer.param_groups[0]["lr"]
    print(f" Current learning rate: {current_lr:.6f}")

    print("TRAINING PHASE:")
    train_loss, train_acc = train_epoch(
        model, train_loader, criterion, optimizer, device
    )

    print("VALIDATION PHASE:")
    val_loss, val_acc = validate(model, val_loader, criterion, device)

    training_losses.append(train_loss)
    train_accs.append(train_acc)
    val_losses.append(val_loss)
    val_accs.append(val_acc)

    print("EPOCH SUMMARY")
    print(f"Training   → Loss: {train_loss:.4f} | Accuracy: {train_acc:.2f}%")
    print(f"Validation → Loss: {val_loss:.4f} | Accuracy: {val_acc:.2f}%")

    if epoch > 0:
        train_loss_change = training_losses[-1] - training_losses[-2]
        val_acc_change = val_accs[-1] - val_accs[-2]
        print(
            f"Changes    → Train Loss: {train_loss_change:+.4f} | Val Acc: {val_acc_change:+.2f}%"
        )

        print("Updating learing rate")
        old_lr = optimizer.param_groups[0]["lr"]
        sheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]["lr"]

        if new_lr < old_lr:
            print(f"Learning rate reduced {old_lr:.6f} - {new_lr:.6f}")

        if val_acc > best_val_acc:
            improvement = val_acc - best_val_acc
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_brain_tumor_model.pth")

    print("TRAINING COMPLETED")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")

    # test evaluation
    def evaluate_model(model, dataloader, device):

        print("Evaluating model on test set...")

        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc="Testing"):
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        print(f"Evaluated {len(all_labels)} test samples")
        return np.array(all_preds), np.array(all_labels)

    print("Evaluating on test set")

    test_preds, test_labels = evaluate_model(model, test_loader, device)

    test_accuracy = 100 * np.sum(test_preds == test_labels) / len(test_labels)
    print(f" Test Accuracy: {test_accuracy:.2f}%")

    class_names = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

    print("Classification Report:")
    print(classification_report(test_labels, test_preds, target_names=class_names))
