import torch
from torch.utils.data import DataLoader
from torchvision import transforms
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

# Test function
def test_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, (digits1, digits2) in test_loader:
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

# Run test
accuracy = test_model(model, test_loader, device)
print(f"Test Accuracy: {accuracy * 100:.2f}%")