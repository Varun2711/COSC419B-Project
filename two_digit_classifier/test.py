import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from collections import defaultdict
from config import get_config
from dataset import AllInOneJerseyNumberDataset
from model import TwoDigitClassifier


def test_model(cfg):
    data_dir = cfg["data_dir"]
    gt_file = cfg["gt_file"]
    batch_size = cfg["batch_size"]
    device = cfg["device"]
    model_arch = cfg["model_arch"]
    saved_model = cfg["saved_model"]

    test_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    print("Loading data from:", "\nimages:", data_dir, "\ngt:", gt_file)
    test_dataset = AllInOneJerseyNumberDataset(
        image_dir=data_dir, gt_file=gt_file, transform=test_transform
    )

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = TwoDigitClassifier(model_arch).to(device)
    model.load_state_dict(torch.load(os.path.join(cfg["model_dir"], saved_model)))
    model.eval()

    print(f"Loaded model {model_arch} from {saved_model}")

    # Tracklet-level test
    accuracy = test_model_grouped(model, test_loader, device)

    print(f"Group Accuracy: {accuracy * 100:.2f}%")


# Image-level test function
def test_model_image(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        test_loop = tqdm(test_loader, desc="Testing", leave=False)
        for images, (digits1, digits2) in test_loop:
            images = images.to(device)
            digits1 = digits1.to(device)
            digits2 = digits2.to(device)

            # Forward pass
            pred1, pred2 = model(images)

            # Get predictions
            pred1 = torch.argmax(pred1, dim=1)
            pred2 = torch.argmax(pred2, dim=1)

            # Compare with ground truth
            correct += ((pred1 == digits1) & (pred2 == digits2)).sum().item()
            total += digits1.size(0)

    accuracy = correct / total
    return accuracy


# Tracklet-level test function
def test_model_grouped(model, test_loader, device):
    model.eval()
    group_predictions = defaultdict(list)  # Store predictions for each group
    group_labels = {}  # Store ground truth labels for each group

    # Iterate through the test dataset
    with torch.no_grad():
        test_loop = tqdm(test_loader, desc="Testing", leave=False)
        for images, (digits1, digits2), group_ids in test_loop:
            images = images.to(device)
            digits1 = digits1.to(device)
            digits2 = digits2.to(device)

            # Forward pass
            pred1, pred2 = model(images)

            # Get predictions
            pred1 = torch.argmax(pred1, dim=1).cpu().numpy()  # First digit
            pred2 = torch.argmax(pred2, dim=1).cpu().numpy()  # Second digit

            # Combine predictions into jersey numbers
            for i, (d1, d2) in enumerate(zip(pred1, pred2)):
                if d2 == 10:  # "empty" class
                    pred_number = str(d1)
                else:
                    pred_number = f"{d1}{d2}"

                # Store prediction for the group
                group_predictions[group_ids[i]].append(pred_number)

            # Store ground truth labels for the group
            for i, group_id in enumerate(group_ids):
                if group_id not in group_labels:
                    if digits2[i].item() == 10:  # Single-digit
                        group_labels[group_id] = str(digits1[i].item())
                    else:  # Double-digit
                        group_labels[group_id] = (
                            f"{digits1[i].item()}{digits2[i].item()}"
                        )

    # Perform majority voting for each group
    correct = 0
    total = 0
    for group_id, predictions in group_predictions.items():
        # Get the most frequent prediction
        from collections import Counter

        voted_prediction = Counter(predictions).most_common(1)[0][0]

        # Compare with ground truth
        if voted_prediction == group_labels[group_id]:
            correct += 1
        total += 1

    # Calculate group accuracy
    accuracy = correct / total
    return accuracy


if __name__ == "__main__":
    cfg = get_config(require_mode=False, default_mode="test")
    test_model(cfg)
