import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

import matplotlib.pyplot as plt

from config import get_config, update_config
from dataset import AllInOneJerseyNumberDataset
from model import TwoDigitClassifier


def train_model(cfg):
    data_dir = cfg["data_dir"]
    gt_file = cfg["gt_file"]
    batch_size = cfg["batch_size"]
    device = cfg["device"]

    # Create model directory if needed
    os.makedirs(cfg["model_dir"], exist_ok=True)

    train_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    denormalize = transforms.Compose([
        transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
            std=[1/0.229, 1/0.224, 1/0.225]
        ),
        transforms.ToPILImage()
    ])

    # Split dataset
    print("Loading data from:", "\nimages:", data_dir, "\ngt:", gt_file)
    full_dataset = AllInOneJerseyNumberDataset(
        image_dir=data_dir, gt_file=gt_file, transform=train_transform
    )
    train_samples = []
    val_samples = []
    for sample in full_dataset.samples:
        filename = os.path.basename(sample[0])
        prefix = int(filename.split('_')[0])
        if prefix <= 800:
            train_samples.append(sample)
        else:
            val_samples.append(sample)

    train_dataset = torch.utils.data.Subset(full_dataset, indices=range(len(train_samples)))
    val_dataset = torch.utils.data.Subset(full_dataset, indices=range(len(train_samples), len(train_samples)+len(val_samples)))

    print(f"Done. Training set: {len(train_dataset)}, Validation set: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        prefetch_factor=2,
    )

    model = TwoDigitClassifier(cfg["model_arch"]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    print("Built model:", cfg["model_arch"])

    # Training loop with validation
    train_losses = []
    val_losses = []
    accuracies = []

    for epoch in range(cfg["epochs"]):
        model.train()
        running_loss = 0.0

        train_loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{cfg["epochs"]}")
        for images, (digits1, digits2), _, _ in train_loop:
            images = images.to(device)
            digits1 = digits1.to(device)
            digits2 = digits2.to(device)

            pred1, pred2 = model(images)
            loss1 = criterion(pred1, digits1)
            loss2 = criterion(pred2, digits2)
            total_loss = loss1 + loss2

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()

        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        sample_data = []    # Store (image,true, pred) tuples

        with torch.no_grad():
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=4,
                prefetch_factor=2,
            )
            for images, (d1, d2), _, filename in tqdm(val_loader, desc="Validating", leave=False):
                images = images.to(device)
                d1 = d1.to(device)
                d2 = d2.to(device)

                # Forward pass
                pred1, pred2 = model(images)

                # Calculate loss
                loss1 = criterion(pred1, d1)
                loss2 = criterion(pred2, d2)
                total_loss = loss1 + loss2
                val_loss += total_loss.item() * images.size(0)

                # Get predictions
                pred1_labels = torch.argmax(pred1, dim=1)
                pred2_labels = torch.argmax(pred2, dim=1)

                # Count correct predictions
                correct += ((pred1_labels == d1) & (pred2_labels == d2)).sum().item()
                total += images.size(0)

                # Store sample data for visualization
                for i in range(len(images)):
                    sample_data.append((
                        images[i].cpu(),
                        f"{d1[i].item()}{d2[i].item() if d2[i] != 10 else ''}",
                        f"{pred1_labels[i].item()}{pred2_labels[i].item() if pred2_labels[i] != 10 else ''}"
                    ))

        # Calculate validation metrics
        avg_val_loss = val_loss / total
        val_accuracy = correct / total

        # Save metrics
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)
        val_losses.append(avg_val_loss)
        accuracies.append(val_accuracy)

        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader):.4f}, Validation Accuracy: {val_accuracy * 100:.2f}%")

        # Create sample visualization
        correct_samples = [s for s in sample_data if s[1] == s[2]]
        wrong_samples = [s for s in sample_data if s[1] != s[2]]

        plt.ioff()
        plt.close('all')
        
        # Create visualization plot
        plt.figure(figsize=(15, 12))
        plt.suptitle(f"Validation Samples - Epoch {epoch+1}", y=0.95)
        
        # Plot correct samples
        for i in range(min(10, len(correct_samples))):
            plt.subplot(4, 5, i+1)
            img = denormalize(correct_samples[i][0])
            plt.imshow(img)
            plt.title(correct_samples[i][1], color='green')
            plt.axis('off')
            
        # Plot wrong samples
        for i in range(min(10, len(wrong_samples))):
            plt.subplot(4, 5, i+11)
            img = denormalize(wrong_samples[i][0])
            title = f"{wrong_samples[i][1]}→{wrong_samples[i][2]}"
            plt.imshow(img)
            plt.title(title, color='red')
            plt.axis('off')
            
        # Save visualization
        vis_path = os.path.join(
            cfg["model_dir"],
            f"{cfg['model_arch']}_bs{cfg['batch_size']}_epoch{epoch+1}.jpg"
        )
        plt.savefig(vis_path)
        plt.close()

    # Save model
    model_name = f"{cfg['model_arch']}_bs{cfg['batch_size']}_epoch{cfg['epochs']}.pth"
    save_path = os.path.join(cfg["model_dir"], model_name)
    torch.save(model.state_dict(), save_path)

    print(f"Model saved to {save_path}")

    # Update config
    update_config("default_model", model_name)

    # Plot results
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Validation")
    plt.title(f"Loss ({cfg['model_arch']}, bs={cfg['batch_size']})")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(accuracies)
    plt.title(f"Accuracy ({cfg['model_arch']}, bs={cfg['batch_size']})")

    fig_path = os.path.join(
        cfg["model_dir"],
        f"{cfg['model_arch']}_bs{cfg['batch_size']}_epoch{cfg['epochs']}_metrics.png"
    )
    plt.savefig(fig_path)

    print(f"Training figure saved to {fig_path}")


if __name__ == "__main__":
    cfg = get_config(require_mode=False, default_mode="train")
    train_model(cfg)
