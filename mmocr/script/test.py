import os, json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
from tqdm import tqdm
from dataset import JerseyDataset
from collections import defaultdict

# Import the same model components used in training
from mmocr.models.textrecog.recognizers import CRNN
from mmocr.models.textrecog.postprocessors import CTCPostProcessor

# Load the trained model
def load_model(model_path, device):
    # Recreate the model architecture (same as in training)
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
    
    # Add the same post-processing layers as in training
    model.backbone = nn.Sequential(
        model.backbone,
        nn.AdaptiveAvgPool2d((1, None)),
    )
    
    model = model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

# Define the same helper functions as in training
def label_to_text(label_tensor):
    mapping = {i: str(i) for i in range(10)}
    mapping[10] = ""
    return "".join(mapping[x] for x in label_tensor.tolist())

class SimpleDataSample:
    def __init__(self, gt_text, valid_ratio=1.0):
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

# Define the same collate function as in training
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

# Define the same transform as in training
class ResizeToFixedHeight(object):
    def __init__(self, target_height):
        self.target_height = target_height

    def __call__(self, img):
        w, h = img.size
        new_width = int(w * self.target_height / h)
        return img.resize((new_width, self.target_height), Image.BILINEAR)

test_transform = transforms.Compose([
    ResizeToFixedHeight(32),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def evaluate_model(model, dataloader, device, gt_json_path, ignore_threshold=0.9):
    ctc_postprocessor = CTCPostProcessor(dictionary=model.decoder.dictionary)
    
    # Load ground truth to get all tracklet IDs and their actual jersey numbers
    with open(gt_json_path, 'r') as f:
        all_gt = json.load(f)
    all_tracklet_ids = set(all_gt.keys())

    # Metrics
    metrics = {
        'digit_correct': 0,
        'digit_total': 0,
        'number_correct': 0,
        'number_total': 0,
        'tracklet_correct': 0,
        'tracklet_total': len(all_tracklet_ids)
    }

    # Storage for predictions and tracklet info
    tracklet_predictions = defaultdict(list)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            imgs = batch['img'].to(device)
            labels = batch['gt_label'].to(device)
            track_nums = batch['track_num']

            data_samples = [SimpleDataSample(label_to_text(labels[i]), valid_ratio=1.0)
                          for i in range(labels.size(0))]
            
            outputs = model(imgs, data_samples=data_samples)
            predicted_samples = ctc_postprocessor(outputs, data_samples=data_samples)
            pred_strings = [sample.pred_text.item for sample in predicted_samples]
            gt_strings = [label_to_text(labels[i]) for i in range(labels.size(0))]

            # Update metrics
            for pred, gt in zip(pred_strings, gt_strings):
                metrics['number_total'] += 1
                if pred == gt:
                    metrics['number_correct'] += 1
                
                # Digit-level comparison
                for p_digit, t_digit in zip(pred.ljust(2), gt.ljust(2)):
                    metrics['digit_total'] += 1
                    if p_digit == t_digit:
                        metrics['digit_correct'] += 1

            # Store predictions by tracklet
            for track_num, pred in zip(track_nums, pred_strings):
                tracklet_predictions[track_num].append(pred)

    # Process tracklet-level predictions with threshold
    tracklet_output = {}
    for track_num in all_tracklet_ids:
        gt_num = int(all_gt.get(track_num, -1))
        preds = tracklet_predictions.get(track_num, [])
        total_preds = len(preds)
        
        # Count empty predictions (considered as -1)
        num_empty = sum(1 for p in preds if not p.strip())
        
        # Apply threshold: if >=90% are empty, final prediction is -1
        if total_preds > 0 and num_empty / total_preds >= ignore_threshold:
            pred_num = -1
        else:
            # Filter out empty predictions when below threshold
            valid_preds = [p for p in preds if p.strip()]
            
            if not valid_preds:
                pred_num = -1
            else:
                # Get most frequent non-empty prediction
                pred_counts = defaultdict(int)
                for p in valid_preds:
                    try:
                        p_num = int(p)
                        if 0 <= p_num < 100:  # Only count valid numbers
                            pred_counts[p] += 1
                    except ValueError:
                        continue
                
                if pred_counts:
                    consensus_pred = max(pred_counts.items(), key=lambda x: x[1])[0]
                    pred_num = int(consensus_pred)
                else:
                    pred_num = -1
        
        tracklet_output[track_num] = pred_num
        
        # Check if correct (compare with actual ground truth)
        if pred_num == gt_num:
            metrics['tracklet_correct'] += 1

    return {
        'number_accuracy': metrics['number_correct'] / metrics['number_total'] if metrics['number_total'] > 0 else 0,
        'digit_accuracy': metrics['digit_correct'] / metrics['digit_total'] if metrics['digit_total'] > 0 else 0,
        'tracklet_accuracy': metrics['tracklet_correct'] / metrics['tracklet_total'] if metrics['tracklet_total'] > 0 else 0,
        'tracklet_predictions': tracklet_output,
        'details': {
            'correct_numbers': metrics['number_correct'],
            'total_numbers': metrics['number_total'],
            'correct_digits': metrics['digit_correct'],
            'total_digits': metrics['digit_total'],
            'correct_tracklets': metrics['tracklet_correct'],
            'total_tracklets': metrics['tracklet_total'],
            'tracklet_stats': {
                'total_tracklets_with_predictions': len(tracklet_predictions),
                'tracklets_forced_to_empty': sum(1 for tn in all_tracklet_ids 
                                              if len(tracklet_predictions.get(tn, [])) > 0 and 
                                              sum(1 for p in tracklet_predictions.get(tn, []) if not p.strip()) / 
                                              len(tracklet_predictions.get(tn, [])) >= ignore_threshold)
            }
        }
    }

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_model("../model/crnn_mini_finetuned_5start.pth", device)
    
    # Configuration
    test_image_dir = "../data/crops/test"
    test_gt_json = "../data/test_gt.json"
    output_json_path = "../crnn_output.json"
    
    # Prepare dataset
    test_dataset = JerseyDataset(img_dir=test_image_dir, gt_json=test_gt_json, transform=test_transform)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=custom_collate_fn)
    
    # Evaluate
    results = evaluate_model(model, test_dataloader, device, test_gt_json)
    
    # Print results
    print("\nEvaluation Results:")
    print(f"Number-level Accuracy: {results['number_accuracy'] * 100:.2f}%")
    print(f"Digit-level Accuracy: {results['digit_accuracy'] * 100:.2f}%")
    print(f"Tracklet-level Accuracy: {results['tracklet_accuracy'] * 100:.2f}%")
    print(f"Correct Numbers: {results['details']['correct_numbers']}/{results['details']['total_numbers']}")
    print(f"Correct Digits: {results['details']['correct_digits']}/{results['details']['total_digits']}")
    print(f"Correct Tracklets: {results['details']['correct_tracklets']}/{results['details']['total_tracklets']}")
    
    # Save predictions
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, 'w') as f:
        json.dump(results['tracklet_predictions'], f, indent=2)
    print(f"\nPredictions saved to {output_json_path}")

if __name__ == "__main__":
    main()

