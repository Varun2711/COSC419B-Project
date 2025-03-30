import os
import json
import re
from collections import defaultdict
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

# Import model components (same as in test.py)
from mmocr.models.textrecog.recognizers import CRNN
from mmocr.models.textrecog.postprocessors import CTCPostProcessor

# Reuse the same model loading function from test.py
def load_model(model_path, device):
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
    
    model.backbone = nn.Sequential(
        model.backbone,
        nn.AdaptiveAvgPool2d((1, None)),
    )
    
    model = model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

# Helper classes (same as in test.py)
class SimpleDataSample:
    def __init__(self, gt_text="", valid_ratio=1.0):
        self._gt_text = DummyText(gt_text)
        self.valid_ratio = valid_ratio
        self.pred_text = None

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

def label_to_text(label_tensor):
    mapping = {i: str(i) for i in range(10)}
    mapping[10] = ""
    return "".join(mapping[x] for x in label_tensor.tolist())

# Custom transform (same as in test.py)
class ResizeToFixedHeight:
    def __init__(self, target_height):
        self.target_height = target_height

    def __call__(self, img):
        w, h = img.size
        new_width = int(w * self.target_height / h)
        return img.resize((new_width, self.target_height), Image.BILINEAR)

# Inference transform
inference_transform = transforms.Compose([
    ResizeToFixedHeight(32),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Custom collate function (simplified version)
def inference_collate_fn(batch):
    imgs = [sample['img'] for sample in batch]
    max_width = max(img.shape[2] for img in imgs)
    padded_imgs = []
    
    for img in imgs:
        pad_width = max_width - img.shape[2]
        padded = F.pad(img, (0, pad_width), mode='constant', value=0)
        padded_imgs.append(padded)
    
    return {
        'img': torch.stack(padded_imgs, dim=0),
        'filename': [sample['filename'] for sample in batch],
        'track_num': [sample['track_num'] for sample in batch]
    }

# Dataset class for inference
class InferenceJerseyDataset:
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.samples = []
        
        for fname in os.listdir(img_dir):
            if fname.endswith('.jpg'):
                # Extract track number using regex
                m = re.match(r"(\d+)_.*\.jpg", fname)
                if m:
                    track_num = m.group(1)
                    self.samples.append({
                        'img_path': os.path.join(img_dir, fname),
                        'filename': fname,
                        'track_num': track_num
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = Image.open(sample['img_path']).convert("RGB")
        
        if self.transform:
            img = self.transform(img)
            
        return {
            'img': img,
            'filename': sample['filename'],
            'track_num': sample['track_num']
        }

def run_inference(model, dataloader, device, min_track=0, max_track=1425, ignore_threshold=0.9):
    ctc_postprocessor = CTCPostProcessor(dictionary=model.decoder.dictionary)
    tracklet_predictions = defaultdict(list)
    image_predictions = {}
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Processing Images"):
            imgs = batch['img'].to(device)
            filenames = batch['filename']
            track_nums = batch['track_num']
            
            data_samples = [SimpleDataSample() for _ in range(len(filenames))]
            
            outputs = model(imgs, data_samples=data_samples)
            predicted_samples = ctc_postprocessor(outputs, data_samples=data_samples)
            
            for filename, track_num, sample in zip(filenames, track_nums, predicted_samples):
                pred = sample.pred_text.item
                tracklet_predictions[track_num].append(pred)
                image_predictions[filename] = pred
    
    complete_output = {}
    for track_num in range(min_track, max_track + 1):
        str_track_num = str(track_num)
        if str_track_num in tracklet_predictions:
            preds = tracklet_predictions[str_track_num]
            total_preds = len(preds)
            num_empty = sum(1 for p in preds if not p.strip())
            
            if total_preds > 0 and num_empty / total_preds >= ignore_threshold:
                complete_output[str_track_num] = -1
            else:
                valid_preds = [p for p in preds if p.strip()]
                if not valid_preds:
                    complete_output[str_track_num] = -1
                else:
                    pred_counts = defaultdict(int)
                    for p in valid_preds:
                        try:
                            p_num = int(p)
                            if 0 <= p_num < 100:
                                pred_counts[p] += 1
                        except ValueError:
                            continue
                    
                    if pred_counts:
                        consensus_pred = max(pred_counts.items(), key=lambda x: x[1])[0]
                        complete_output[str_track_num] = int(consensus_pred)
                    else:
                        complete_output[str_track_num] = -1
        else:
            complete_output[str_track_num] = -1
    
    return {
        'image_predictions': image_predictions,
        'tracklet_predictions': complete_output,
        'stats': {
            'total_images': len(image_predictions),
            'total_tracklets_in_range': max_track - min_track + 1,
            'tracklets_with_predictions': len(tracklet_predictions),
            'tracklets_missing': (max_track - min_track + 1) - len(tracklet_predictions),
            'tracklets_with_empty_consensus': sum(1 for p in complete_output.values() if p == -1),
            'tracklets_with_valid_consensus': sum(1 for p in complete_output.values() if p != -1)
        }
    }

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_model("../model/crnn_mini_finetuned_5start.pth", device)
    
    challenge_dir = "../data/crops/challenge"
    output_json_path = "../challenge_predictions.json"
    
    dataset = InferenceJerseyDataset(img_dir=challenge_dir, transform=inference_transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=inference_collate_fn)
    
    results = run_inference(model, dataloader, device)
    
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, 'w') as f:
        json.dump(results['tracklet_predictions'], f, indent=2)
    
    # Updated print statements to match the actual stats keys
    print(f"\nInference complete. Results saved to {output_json_path}")
    print(f"Processed {results['stats']['total_images']} images")
    print(f"Tracklets in range [0, 1425]: {results['stats']['total_tracklets_in_range']}")
    print(f"Tracklets with images: {results['stats']['tracklets_with_predictions']}")
    print(f"Tracklets missing: {results['stats']['tracklets_missing']}")
    print(f"Tracklets with valid predictions: {results['stats']['tracklets_with_valid_consensus']}")
    print(f"Tracklets marked empty: {results['stats']['tracklets_with_empty_consensus']}")

if __name__ == "__main__":
    main()