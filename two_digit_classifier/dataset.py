import os
import json
from PIL import Image
from torch.utils.data import Dataset


class StructuredJerseyNumberDataset(Dataset):
    def __init__(self, image_dir, gt_file, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.samples = []

        # Load ground truth from JSON
        with open(gt_file, "r") as f:
            self.gt = json.load(f)

        # Iterate through folders and collect valid samples
        for folder_name, label in self.gt.items():
            if label == -1:  # Skip invalid labels
                continue
            folder_path = os.path.join(image_dir, folder_name)
            for img_name in os.listdir(folder_path):
                img_path = os.path.join(folder_path, img_name)
                self.samples.append((img_path, label, folder_name))  # Add group ID

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, group_id = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # Convert label to two-digit format
        if label < 10:  # Single-digit
            digit1 = label
            digit2 = 10  # "empty" class
        else:  # Double-digit
            digit1 = label // 10
            digit2 = label % 10

        return image, (digit1, digit2), group_id


class AllInOneJerseyNumberDataset(Dataset):
    def __init__(self, image_dir, gt_file, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.samples = []

        # Load ground truth from JSON
        with open(gt_file, "r") as f:
            self.gt = json.load(f)

        # Iterate through all images in the folder
        for img_name in os.listdir(image_dir):
            img_path = os.path.join(image_dir, img_name)
            group_id = img_name.split("_")[0]  # Extract group ID from filename
            label = self.gt.get(group_id, -1)  # Get label from JSON

            if label == -1:  # Skip invalid labels
                continue

            self.samples.append((img_path, label, group_id))  # Add group ID

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, group_id = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # Convert label to two-digit format
        if label < 10:  # Single-digit
            digit1 = label
            digit2 = 10  # "empty" class
        else:  # Double-digit
            digit1 = label // 10
            digit2 = label % 10

        return image, (digit1, digit2), group_id