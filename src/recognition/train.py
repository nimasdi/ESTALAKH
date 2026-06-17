import os

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.models import EfficientNet
from torchvision.transforms import transforms

from src.recognition.model import load_base_model


class Chars74KDigits1to9(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        for label in range(2, 11):
            if label < 10:
                label_dir = os.path.join(self.root_dir, f"Sample00{label}")
            else:
                label_dir = os.path.join(self.root_dir, f"Sample0{label}")
            for filename in os.listdir(label_dir):
                if filename.endswith((".jpg", ".png")):
                    self.image_paths.append(os.path.join(label_dir, filename))
                    self.labels.append(label - 2)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def train(model: EfficientNet, loader: DataLoader, criterion, optimizer, device):
    model.train()
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()

    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        predicted = outputs.argmax(dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


def validate(model: EfficientNet, loader: DataLoader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            predicted = outputs.argmax(dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    return running_loss / total, correct / total


def main():
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    dataset = Chars74KDigits1to9("../../data/English/Fnt", transform=transform)
    dataset_length = len(dataset)
    train_size = int(0.8 * dataset_length)
    val_size = dataset_length - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_base_model()
    model.to(device)

    for param in model.features.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()

    criterion = nn.CrossEntropyLoss()
    optimizer_head = torch.optim.Adam(model.classifier.parameters(), lr=1e-3)

    epochs_head = 8
    for epoch in range(epochs_head):
        train_loss, train_acc = train(
            model, train_loader, criterion, optimizer_head, device
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        print(
            f"Epoch {epoch + 1}/{epochs_head} - "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

    for param in model.features[-3].parameters():
        param.requires_grad = True

    optimizer_finetune = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5
    )

    epochs_finetune = 10
    best_acc = 0
    for epoch in range(epochs_finetune):
        train_loss, train_acc = train(
            model, train_loader, criterion, optimizer_finetune, device
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        print(
            f"Epoch {epoch + 1}/{epochs_finetune} - "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "../models/best_model.pth")


if __name__ == "__main__":
    main()
