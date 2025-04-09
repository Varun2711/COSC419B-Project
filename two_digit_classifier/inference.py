import os
import json
import torch
from torchvision import transforms
from PIL import Image
from model import TwoDigitClassifier
from config import get_config
from tqdm import tqdm

def run_inference(cfg):
    # Initialize model
    device = cfg["device"]
    device = torch.device("cpu")
    print(f"Using device: {device}")
    model = TwoDigitClassifier(cfg["model_arch"]).to(device)
    model.load_state_dict(torch.load(os.path.join(cfg["model_dir"], cfg["saved_model"])), map_location=device)
    model.eval()
    print(f"Loaded model {cfg['model_arch']} from {cfg['saved_model']}")

    # Image preprocessing
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Process challenge directory
    image_dir = "../data/Preprocessing/Grayscale/challenge"
    output_file = "out_challenge.json"
    predictions = {str(i): -1 for i in range(1426)}  # Initialize all groups as -1

    # Process each image in the directory
    for img_name in tqdm(os.listdir(image_dir)):
        if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        img_path = os.path.join(image_dir, img_name)
        group_id = img_name.split('_')[0]  # Extract group ID from filename

        try:
            # Load and preprocess image
            image = Image.open(img_path).convert('RGB')
            image = transform(image).unsqueeze(0).to(device)

            # Run inference
            with torch.no_grad():
                logits1, logits2 = model(image)
                probs1 = torch.softmax(logits1, dim=1)
                probs2 = torch.softmax(logits2, dim=1)
                
                pred1 = torch.argmax(probs1, dim=1).item()
                pred2 = torch.argmax(probs2, dim=1).item()
                conf1 = probs1.max().item()
                conf2 = probs2.max().item()

            # Apply confidence threshold (0.6)
            if conf1 >= 0.6 and (pred2 == 10 or conf2 >= 0.6):
                prediction = str(pred1) if pred2 == 10 else f"{pred1}{pred2}"
                predictions[group_id] = int(prediction)
            #     print(f"Processed {img_name}: Predicted {prediction} (Confidence: {conf1:.2f}, {conf2:.2f})")
            # else:
            #     print(f"Processed {img_name}: Low confidence ({conf1:.2f}, {conf2:.2f}) - Skipped")

        except Exception as e:
            print(f"Error processing {img_name}: {str(e)}")
            continue

    # Save predictions
    with open(output_file, 'w') as f:
        json.dump(predictions, f, indent=2)
    print(f"\nPredictions saved to {output_file}")

if __name__ == "__main__":
    cfg = get_config(require_mode=False, default_mode="inference")
    
    # Update config for inference-specific settings
    cfg.update({
        "data_dir": "../data/Preprocessing/Grayscale/challenge",
        "output_json": "out_challenge.json",
        # Add any other inference-specific configs here
    })
    
    run_inference(cfg)