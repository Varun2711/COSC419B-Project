import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from collections import defaultdict
from dataset import StructuredJerseyNumberDataset, AllInOneJerseyNumberDataset
from model import TwoDigitClassifier

# Define transforms for test data
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load test dataset (choose one of the two methods)
# Method 1: Structured folders
# test_dataset = StructuredJerseyNumberDataset(
#     image_dir="/data/test/images",
#     gt_file="/data/test/test_gt.json",
#     transform=test_transform
# )

# Method 2: All-in-one folder
test_dataset = AllInOneJerseyNumberDataset(
    image_dir="../out/SoccerNetResults/crops_test/imgs",
    gt_file="../data/SoccerNet/test/test_gt.json",
    transform=test_transform
)

# Create DataLoader
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")
model = TwoDigitClassifier().to(device)
model.load_state_dict(torch.load("two_digit_classifier.pth"))

# Image-level test function
def test_model(model, test_loader, device):
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
        for images, (digits1, digits2), group_ids in test_loader:
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
                        group_labels[group_id] = f"{digits1[i].item()}{digits2[i].item()}"

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

# Run test
accuracy = test_model_grouped(model, test_loader, device)
print(f"Group Accuracy: {accuracy * 100:.2f}%")