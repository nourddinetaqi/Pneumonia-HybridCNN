import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix, classification_report

from model import HybridPneumoniaCNN


def get_data_loaders(data_dir=r"chest_xray", batch_size=16, image_size=224, val_split=0.20):
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.05, 0.05),
            scale=(1.0, 1.05)
        ),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor()
    ])

    test_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])

    full_train_dataset = datasets.ImageFolder(f"{data_dir}/train", transform=train_transform)
    num_train = len(full_train_dataset)
    num_val = int(val_split * num_train)
    num_train_final = num_train - num_val

    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [num_train_final, num_val],
        generator=torch.Generator().manual_seed(42)
    )


    test_dataset = datasets.ImageFolder(f"{data_dir}/test", transform=test_transform)

    print("Total original train images:", num_train)
    print("Train Split:", len(train_dataset))
    print("Val split:", len(val_dataset))
    print("Test images:", len(test_dataset))
    print("Class mapping:", full_train_dataset.class_to_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, full_train_dataset.classes



def validate(model, loader, criterion, device):
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
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    return running_loss / total, correct / total



def evaluate(model, loader, device):
    model.eval()

    all_labels= []
    all_preds = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, 
                                   target_names=["NORMAL", "PNEUMONIA"])
    
    return cm, report



def compute_class_weights(train_loader, device):
    subset = train_loader.dataset
    full_dataset = subset.dataset
    class_to_idx = full_dataset.class_to_idx

    normal_counts = 0
    pneumonia_counts = 0
    

    for idx in subset.indices:
        label = full_dataset.targets[idx]

        if label == 0:
            normal_counts += 1
        else:
            pneumonia_counts += 1


    print("\n Training class counts:")
    print("Normal: ", normal_counts)
    print("Pneumonia: ", pneumonia_counts)

    counts = torch.tensor([normal_counts, pneumonia_counts], dtype=torch.float)
    weights = 1.0 / counts
    weights = weights / weights.sum()

    print("computed class weights: ", weights)

    return weights.to(device)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device)

    BATCH_SIZE = 16
    IMAGE_SIZE = 224
    NUM_EPOCHS = 20
    LR = 1e-4
    WEIGHT_DECAY = 1e-4
    PATIENCE = 5
    WEIGHT_PATHS = "best_hybrid_pneumonia_cnn.pth"

    train_loader, val_loader, test_loader, class_names = get_data_loaders(
        batch_size=BATCH_SIZE, image_size=IMAGE_SIZE
    )

    model = HybridPneumoniaCNN(num_classes=2).to(device)
    print("\nModel:")
    print(model)

    class_weights = compute_class_weights(train_loader, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer, mode="min", factor=0.5, patience=2
    )

    best_val_loss = float("inf")
    best_val_acc = 0.0
    epochs_no_improve = 0

    print("\n Staring training....")
    start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        model.train()

        running_loss = 0.0
        correct= 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        
        train_loss = running_loss / total
        train_acc = correct / total

        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}/{NUM_EPOCHS}"
              f"Train loss: {train_loss: .4f}, Train Acc: {train_acc: .4f} | ",
              f"Val loss: {val_loss: .4f}, Val Acc: {val_acc: .4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), WEIGHT_PATHS)
            print(f"Saved new best model (val loss={val_loss: .4f}, val acc: {val_acc: .4f})")

        else:
            epochs_no_improve += 1
            print(f"No, improvement for {epochs_no_improve} epochs.")

        if epochs_no_improve >= PATIENCE:
            print(f"\nEarly stopping triggered at epoch {epoch + 1}.")
            break

        print(f"\nBest Validation Accuracy: {best_val_acc: .4f}")

    end_time = time.time()
    total_time = end_time - start_time
    print("\nTime Taken: ", total_time)
    model.load_state_dict(torch.load(WEIGHT_PATHS, map_location=device))
    cm, report = evaluate(model, test_loader, device)
    print("\nConfusion Matrix: ", cm)
    print("\nClassification Report:\n", report)


if __name__ == "__main__":
    main()