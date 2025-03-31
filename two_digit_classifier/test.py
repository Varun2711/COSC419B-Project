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
    output_json = cfg.get("output_json", "cnn_output.json")
    lc_file = cfg.get("lc_file", "../out/SoccerNetResults/legible.json")

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

    # Load ground truth
    with open(gt_file, "r") as f:
        gt = json.load(f)

    # Load legibility classifier results (if available)
    lc_dict = None
    if "lc_file" in cfg:
        with open(lc_file, "r") as f:
            lc = json.load(f)
            if isinstance(lc, list):
                lc_dict = {item.split("_")[0]: 1 for item in lc}
            else:
                lc_dict = lc
        print(f"Loaded LC results from {cfg['lc_file']}")

    # Tracklet-level test with LC handling
    accuracy, test_res_dict = test_model_grouped(
        model, test_loader, device, gt, lc_dict, confidence_threshold=0.6
    )
    print(f"Tracklet-level Accuracy: {accuracy * 100:.2f}%")

    # Save output JSON
    with open(output_json, "w") as f:
        json.dump(test_res_dict, f)
    print(f"Predictions saved to {output_json}")

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
def test_model_grouped(model, test_loader, device, gt, lc_dict=None, confidence_threshold=0.6):
    model.eval()
    group_scores = defaultdict(lambda: defaultdict(float))
    group_labels = {}
    test_res_dict = {k: -1 for k in gt.keys()}  # Initialize with all GT keys

    with torch.no_grad():
        test_loop = tqdm(test_loader, desc="Testing", leave=False)
        for images, (digits1, digits2), group_ids, _ in test_loop:
            images = images.to(device)
            digits1 = digits1.to(device)
            digits2 = digits2.to(device)

            logits1, logits2 = model(images)
            probs1 = torch.softmax(logits1, dim=1)
            probs2 = torch.softmax(logits2, dim=1)
            
            pred1 = torch.argmax(probs1, dim=1).cpu().numpy()
            pred2 = torch.argmax(probs2, dim=1).cpu().numpy()
            conf1 = probs1.max(dim=1).values.cpu().numpy()
            conf2 = probs2.max(dim=1).values.cpu().numpy()

            for i in range(len(pred1)):
                group_id = group_ids[i]
                is_single_digit = (pred2[i] == 10)
                digit_conf1 = conf1[i]
                digit_conf2 = 1.0 if is_single_digit else conf2[i]

                # Skip if confidence is too low
                if digit_conf1 < confidence_threshold or (not is_single_digit and digit_conf2 < confidence_threshold):
                    continue

                # Format prediction
                pred_number = str(pred1[i]) if is_single_digit else f"{pred1[i]}{pred2[i]}"
                group_scores[group_id][pred_number] += (digit_conf1 * digit_conf2)

                # Store ground truth once
                if group_id not in group_labels:
                    gt_digit1 = digits1[i].item()
                    gt_digit2 = digits2[i].item()
                    group_labels[group_id] = str(gt_digit1) if gt_digit2 == 10 else f"{gt_digit1}{gt_digit2}"

    # Calculate accuracy with LC false positive penalty
    correct = 0
    total = len(gt)
    for group_id, gt_label in gt.items():
        pred = test_res_dict[group_id]
        
        # Case 1: LC wrongly passed an illegible jersey (GT = -1)
        if gt_label == -1:
            if lc_dict and group_id in lc_dict:  # LC marked as legible (false positive)
                correct -= 1  # Penalize
            continue  # Skip further checks for GT = -1

        # Case 2: Normal comparison for legible jerseys
        if str(pred) == str(gt_label):
            correct += 1

        # Update output dict
        if group_id in group_scores and group_scores[group_id]:
            voted_pred = max(group_scores[group_id].items(), key=lambda x: x[1])[0]
            test_res_dict[group_id] = int(voted_pred)

    accuracy = max(correct, correct) / total  # Avoid negative accuracy
    return accuracy, test_res_dict

if __name__ == "__main__":
    cfg = get_config(require_mode=False, default_mode="test")
    test_model(cfg)
