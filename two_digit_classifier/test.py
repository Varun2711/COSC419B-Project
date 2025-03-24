import os, json
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

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=4, prefetch_factor=2)

    model = TwoDigitClassifier(model_arch).to(device)
    model.load_state_dict(torch.load(os.path.join(cfg["model_dir"], saved_model)))
    model.eval()

    print(f"Loaded model {model_arch} from {saved_model}")

    # Tracklet-level test
    accuracy = test_model_grouped(model, test_loader, device)
    print(f"Tracklet-level Accuracy: {accuracy * 100:.2f}%")

    # Image-level test
    # accuracy = test_model_image(model, test_loader, device)
    # print(f"Image-level Accuracy: {accuracy * 100:.2f}%")


# Image-level test function
def test_model_image(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        test_loop = tqdm(test_loader, desc="Testing", leave=False)
        for images, (digits1, digits2), _, _ in test_loop:
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
def test_model_grouped(model, test_loader, device, confidence_threshold=0.6):
    model.eval()
    group_scores = defaultdict(lambda: defaultdict(float))  # {group_id: {pred_number: total_confidence}}
    group_labels = {}  # Store ground truth labels for each group

    with torch.no_grad():
        test_loop = tqdm(test_loader, desc="Testing", leave=False)
        for images, (digits1, digits2), group_ids, _ in test_loop:
            images = images.to(device)
            digits1 = digits1.to(device)
            digits2 = digits2.to(device)

            # Forward pass
            logits1, logits2 = model(images)
            
            # Convert logits to probabilities
            probs1 = torch.softmax(logits1, dim=1)
            probs2 = torch.softmax(logits2, dim=1)
            
            # Get predictions and confidences
            pred1 = torch.argmax(probs1, dim=1).cpu().numpy()
            pred2 = torch.argmax(probs2, dim=1).cpu().numpy()
            conf1 = probs1.max(dim=1).values.cpu().numpy()
            conf2 = probs2.max(dim=1).values.cpu().numpy()

            for i in range(len(pred1)):
                # Handle empty second digit case
                is_single_digit = (pred2[i] == 10)
                
                # Calculate effective confidence
                digit_conf1 = conf1[i]
                digit_conf2 = 1.0 if is_single_digit else conf2[i]
                
                # Confidence filtering
                if digit_conf1 < confidence_threshold or \
                   (not is_single_digit and digit_conf2 < confidence_threshold):
                    continue  # Skip low-confidence predictions
                
                # Calculate weighted confidence score
                total_conf = digit_conf1 * (digit_conf2 if not is_single_digit else 1.0)
                
                # Format prediction
                if is_single_digit:
                    pred_number = str(pred1[i])
                else:
                    pred_number = f"{pred1[i]}{pred2[i]}"
                
                # Store confidence score for this prediction
                group_scores[group_ids[i]][pred_number] += total_conf

                # Store ground truth (only need to do this once per group)
                if group_ids[i] not in group_labels:
                    if digits2[i].item() == 10:  # Single-digit label
                        group_labels[group_ids[i]] = str(digits1[i].item())
                    else:
                        group_labels[group_ids[i]] = f"{digits1[i].item()}{digits2[i].item()}"

    # Calculate accuracy
    correct = 0
    total = 0
    test_res_dict = {}
    for group_id, scores in group_scores.items():
        if not scores:  # No valid predictions after filtering
            voted_prediction = "-1"
        else:
            # Get prediction with highest accumulated confidence
            voted_prediction = max(scores.items(), key=lambda x: x[1])[0]
        
        if voted_prediction == group_labels.get(group_id, "-1"):
            correct += 1
        total += 1
        test_res_dict[group_id] = int(voted_prediction)

    with open("output_0.005.json", "w") as f:
        json.dump(test_res_dict, f)
    accuracy = correct / total if total > 0 else 0.0
    return accuracy


if __name__ == "__main__":
    cfg = get_config(require_mode=False, default_mode="test")
    test_model(cfg)
