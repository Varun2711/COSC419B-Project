import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import StructuredJerseyNumberDataset, AllInOneJerseyNumberDataset
from model import TwoDigitClassifier

# Define transforms
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load dataset (choose one of the two methods)
# Method 1: Structured folders
train_dataset = StructuredJerseyNumberDataset(
    image_dir="/data/train/images",
    gt_file="/data/train/train_gt.json",
    transform=train_transform
)

# Method 2: All-in-one folder
# train_dataset = AllInOneJerseyNumberDataset(
#     image_dir="/data/train/images",
#     gt_file="/data/train/train_gt.json",
#     transform=train_transform
# )

# Create DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Initialize model, loss, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")
model = TwoDigitClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training loop
for epoch in range(10):
    model.train()
    running_loss = 0.0
    
    for images, (digits1, digits2) in train_loader:
        images = images.to(device)
        digits1 = digits1.to(device)
        digits2 = digits2.to(device)
        
        # Forward pass
        pred1, pred2 = model(images)
        
        # Calculate losses
        loss1 = criterion(pred1, digits1)
        loss2 = criterion(pred2, digits2)
        total_loss = loss1 + loss2
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        running_loss += total_loss.item()
    
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")

# Save the trained model
torch.save(model.state_dict(), "two_digit_classifier.pth")