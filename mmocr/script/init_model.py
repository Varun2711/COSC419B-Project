import os
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.signal import max_len_seq
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
from tqdm import tqdm

# Import your dataset (JerseyDataset defined in dataset.py)
from dataset import JerseyDataset

# -----------------------------
# Custom Two-Head Decoder, Loss, Postprocessor, and Config
# (same as defined previously)
# -----------------------------
from mmocr.models.textrecog.recognizers import CRNN
from mmocr.models.textrecog.module_losses.ctc_module_loss import CTCModuleLoss
from mmocr.models.textrecog.postprocessors import CTCPostProcessor


# -----------------------------
# Build the Model
# -----------------------------
# For demonstration, we build the CRNN recognizer manually.
# (If available, you can use MMOCR's build_model function.)

# Instantiate the model using our custom decoder.
model = CRNN(
    data_preprocessor=None,
    preprocessor=None,
    backbone=dict(
        type='ResNet31OCR',
        base_channels=3,
        layers=[1, 2, 5, 3],
        channels=[64, 128, 256, 256, 512, 512, 512],
        stage4_pool_cfg=dict(kernel_size=(2, 1), stride=(2, 1)),
        last_stage_pool=False
    ),
    encoder=None,
    # Replace the default decoder with our custom two-head decoder.
    decoder=dict(
        type='CRNNDecoder',
        max_seq_len=2,
        in_channels=512,
        rnn_flag=True,
        module_loss=dict(type='CTCModuleLoss', letter_case='lower'),
        postprocessor=dict(type='CTCPostProcessor'),
        dictionary=dict(
            type='Dictionary',
            dict_file='Dictionary/digits_empty.txt',
            with_padding=True
        )
    )
)

# Define a mapping from integer to string.
def label_to_text(label_tensor):
    # Assuming label_tensor is a 1D tensor with 2 integers.
    mapping = {i: str(i) for i in range(10)}
    mapping[10] = ""
    return "".join(mapping[x] for x in label_tensor.tolist())

class SimpleDataSample:
    def __init__(self, gt_text, valid_ratio=1.0):
        self._gt_text = DummyText(gt_text)  # Ground-truth text wrapped in DummyText.
        self.valid_ratio = valid_ratio
        self.pred_text = None  # This will be updated by CTCPostProcessor.
    @property
    def gt_text(self):
        return self._gt_text
    def get(self, key, default=None):
        if key == 'gt_text':
            return self._gt_text
        return getattr(self, key, default)

class DummyText:
    def __init__(self, text):
        self._text = text
    @property
    def item(self):
        return self._text

# -----------------------------
# Load Pre-trained Weights and Fine Tune
# -----------------------------
# Load the checkpoint from "../model/crnn_mini.pth". Since our custom decoder is different,
# we use strict=False to ignore mismatches.
checkpoint_path = os.path.join('..', 'model', 'crnn_mini_5start.pth')
checkpoint = torch.load(checkpoint_path, map_location='cpu')

# After building your model and before loading weights, add:
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
model = model.to(device)

model.backbone = nn.Sequential(
    model.backbone,
    nn.AdaptiveAvgPool2d((1, None)),  # Pool the height to 1 while keeping width variable
    # nn.Conv2d(1280, 512, kernel_size=1)  # Convert channels from 1280 to 512.
)
model.backbone = model.backbone.to(device)

model.load_state_dict(checkpoint, strict=False)
print("Pre-trained weights loaded.")

# Set up optimizer.
#optimizer = optim.Adam(model.parameters(), lr=1e-4)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

# -----------------------------
# Prepare the Dataset and DataLoader
# -----------------------------
def save_model(model, save_dir="../model", model_name="crnn_mini_finetuned_5start.pth"):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, model_name)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

def custom_collate_fn(batch):
    # Extract images and labels from the batch.
    imgs = [sample['img'] for sample in batch]
    gt_labels = [sample['gt_label'] for sample in batch]
    track_nums = [sample['track_num'] for sample in batch]

    # Find the maximum width in this batch.
    max_width = max(img.shape[2] for img in imgs)  # images are [C, H, W]
    padded_imgs = []
    valid_ratios = []

    for img in imgs:
        original_width = img.shape[2]
        pad_width = max_width - original_width
        # Pad the image on the right.
        padded = F.pad(img, (0, pad_width), mode='constant', value=0)
        padded_imgs.append(padded)
        valid_ratios.append(original_width / max_width)

    return {
        'img': torch.stack(padded_imgs, dim=0),
        'gt_label': torch.stack(gt_labels, dim=0),
        'valid_ratio': torch.tensor(valid_ratios),
        'track_num': track_nums
    }

class ResizeToFixedHeight(object):
    def __init__(self, target_height):
        self.target_height = target_height

    def __call__(self, img):
        # Get current size.
        w, h = img.size
        # Compute new width, preserving aspect ratio.
        new_width = int(w * self.target_height / h)
        return img.resize((new_width, self.target_height), Image.BILINEAR)

train_transform = transforms.Compose(
    [
        ResizeToFixedHeight(32),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

image_directory = r"../data/crops/train"
gt_json_path = r"../data/train_gt.json"
dataset = JerseyDataset(img_dir=image_directory, gt_json=gt_json_path, transform=train_transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=custom_collate_fn)
print("Data loaded.")

# Set model to training mode.
model.train()
loss_module = CTCModuleLoss(letter_case='lower', dictionary=model.decoder.dictionary)
ctc_postprocessor = CTCPostProcessor(dictionary=model.decoder.dictionary)

# -----------------------------
# Dummy Fine-tuning Loop (Updated)
# -----------------------------
num_epochs = 20  # simulate two epochs
for epoch in range(num_epochs):
    epoch_loss = 0.0  # Accumulate loss for this epoch
    num_batches = 0   # Count batches in this epoch

    epoch_correct = 0
    epoch_samples = 0

    for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
        imgs = batch['img'].to(device)
        labels = batch['gt_label'].to(device)  # shape [B, 2] containing (digit1, digit2)

        data_samples = [SimpleDataSample(label_to_text(labels[i]), valid_ratio=batch['valid_ratio'][i].item())
                        for i in range(labels.size(0))]

        outputs = model(imgs, data_samples=data_samples)
        loss_dict = loss_module(outputs, data_samples=data_samples)
        loss = loss_dict['loss_ctc']

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        num_batches += 1

        # Use CTCPostProcessor to decode predictions; this updates each data sample's pred_text attribute.
        predicted_samples = ctc_postprocessor(outputs, data_samples=data_samples)
        # Now extract the predicted text strings:
        pred_strings = [sample.pred_text.item for sample in predicted_samples]
        gt_strings = [label_to_text(labels[i]) for i in range(labels.size(0))]
        correct = sum(1 for p, t in zip(pred_strings, gt_strings) if p == t)
        epoch_correct += correct
        epoch_samples += labels.size(0)

        #print(pred_strings[0:5])

    avg_loss = epoch_loss / num_batches
    epoch_accuracy = epoch_correct / epoch_samples if epoch_samples > 0 else 0
    print(f"Epoch {epoch + 1} - Average Loss: {avg_loss:.4f} - Number-level Accuracy: {epoch_accuracy * 100:.2f}%")
    save_model(model)
