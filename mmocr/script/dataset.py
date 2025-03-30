import os
import json
import re
from PIL import Image
import torch
from torch.utils.data import Dataset


class JerseyDataset(Dataset):
    def __init__(self, img_dir, gt_json, transform=None):
        """
        Args:
            img_dir (str): Directory containing the images.
            gt_json (str): Path to the JSON file with track number labels.
            transform (callable, optional): A function/transform that takes in an image
                and returns a transformed version (e.g., augmentation, normalization).
        """
        self.img_dir = img_dir
        self.transform = transform

        # Load the ground truth labels from JSON.
        with open(gt_json, 'r') as f:
            self.gt_dict = json.load(f)

        self.samples = []
        # Iterate over files in the image directory.
        for fname in os.listdir(img_dir):
            if fname.endswith('.jpg'):
                # Extract track number using regex.
                m = re.match(r"(\d+)_.*\.jpg", fname)
                if m:
                    track_num = m.group(1)
                    # Only consider images with a valid label (i.e. label != -1).
                    if track_num in self.gt_dict:
                        jersey_num = self.gt_dict[track_num]
                        # Convert jersey number into a two-digit tuple.
                        if jersey_num <= -1:
                            #print(track_num)
                            d1, d2 = 10, 10
                        elif 0 <= jersey_num <= 9:
                            # For one-digit jersey, assign the digit and mark second as empty (class 10).
                            d1, d2 = jersey_num, 10
                        else:
                            # For two-digit jersey numbers.
                            d1, d2 = jersey_num // 10, jersey_num % 10
                        sample = {
                            'img_path': os.path.join(img_dir, fname),
                            'label': (d1, d2),
                            'track_num': track_num,
                        }
                        self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = sample['img_path']
        # Load image using OpenCV.
        img = Image.open(img_path).convert("RGB")

        # Apply any given transformation.
        if self.transform:
            img = self.transform(img)

        # Convert label tuple to tensor
        label = torch.tensor(sample['label'], dtype=torch.long)
        # Return a dictionary following the MMOCR-style sample format.
        return {
            'img': img,
            'gt_label': label,  # Tensor of shape [2]
            'filename': os.path.basename(img_path),
            'track_num': sample['track_num'],  # Include track_num in the output
        }


# Example usage:
if __name__ == "__main__":
    # Paths to your data.
    image_directory = r"../data/train"
    gt_json_path = r"../data/train_gt.json"

    # Optionally define a transform (e.g., resize, normalization) using torchvision or mmcv.
    transform = None  # Replace with your transform function if needed.

    dataset = JerseyDataset(img_dir=image_directory, gt_json=gt_json_path, transform=transform)
    print("Number of samples:", len(dataset))

    # Retrieve a sample and print its details.
    sample = dataset[0]
    print("Image path:", sample['filename'])
    print("Ground truth label (first_digit, second_digit):", sample['gt_label'])
