from pathlib import Path
import sys
import os
import argparse

ROOT = './reid/centroids-reid/'
sys.path.append(str(ROOT))  # add ROOT to PATH

import numpy as np
import torch
from tqdm import tqdm
import cv2
from PIL import Image

from config import cfg
from train_ctl_model import CTLModel

from datasets.transforms import ReidTransforms

from torch.utils.data import Dataset, DataLoader

import warnings
warnings.filterwarnings("ignore")

# Based on this repo: https://github.com/mikwieczorek/centroids-reid
# Trained model from here: https://github.com/mikwieczorek/centroids-reid
# Trained model from here: https://drive.google.com/drive/folders/1NWD2Q0JGasGm9HTcOy4ZqsIqK4-IfknK
CONFIG_FILE = str(ROOT+'/configs/256_resnet50.yml')
MODEL_FILE = str(ROOT+'/models/resnet50-19c8e357.pth')

# dict used to get model config and weights using model version
ver_to_specs = {}
ver_to_specs["res50_market"] = (ROOT+'/configs/256_resnet50.yml', ROOT+'/models/market1501_resnet50_256_128_epoch_120.ckpt')
ver_to_specs["res50_duke"]   = (ROOT+'/configs/256_resnet50.yml', ROOT+'/models/dukemtmcreid_resnet50_256_128_epoch_120.ckpt')


def get_specs_from_version(model_version):
    conf, weights = ver_to_specs[model_version]
    conf, weights = str(conf), str(weights)
    return conf, weights

class ImageDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = cv2.imread(img_path)
        img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)
        return img

def generate_features(input_folder, output_folder, model_version='res50_market'):
    print(f"generate_features() input_folder={input_folder}, output_folder={output_folder}")
    # load model
    CONFIG_FILE, MODEL_FILE = get_specs_from_version(model_version)
    cfg.merge_from_file(CONFIG_FILE)
    opts = ["MODEL.PRETRAIN_PATH", MODEL_FILE, "MODEL.PRETRAINED", True, "TEST.ONLY_TEST", True, "MODEL.RESUME_TRAINING", False]
    cfg.merge_from_list(opts)
    model_path = cfg.MODEL.PRETRAIN_PATH
    print(f"generate_features() Ready to loading from {model_path}, cwd {os.getcwd()}")
    model_path = model_path.replace('//', '/')
    print(f"generate_features() Loading from {model_path}, cwd {os.getcwd()}")

    use_cuda = True if torch.cuda.is_available() and cfg.GPU_IDS else False
    model = CTLModel.load_from_checkpoint(model_path, cfg=cfg)

    if use_cuda:
        model.to('cuda')
        print("using GPU")
    else:
        print("using CPU")
    model.eval()
    print(f'centroid_reid: generate_features() evaluation completed!')
    tracks = os.listdir(input_folder)
    transforms_base = ReidTransforms(cfg)
    val_transforms = transforms_base.build_transforms(is_train=False)

    print(f'centroid_reid: generate_features() start to generate features...')
    batch_size = 128  # Adjust based on your GPU memory
    num_workers = 4 # Adjust based on your CPU cores

    with tqdm(total=len(tracks), desc="Processing Tracks") as track_pbar:
        for track in tracks:
            output_file = os.path.join(output_folder, f"{track}_features.npy")
            if os.path.exists(output_file):
                track_pbar.update(1)
                continue

            track_path = os.path.join(input_folder, track)
            images = os.listdir(track_path)
            image_paths = [os.path.join(track_path, img_path) for img_path in images]

            dataset = ImageDataset(image_paths, transform=val_transforms)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

            features = []
            with torch.no_grad():
                for batch in dataloader:
                    input_tensor = batch.cuda() if use_cuda else batch
                    _, global_feat = model.backbone(input_tensor)
                    global_feat = model.bn(global_feat)
                    features.extend(global_feat.cpu().numpy())

            np_feat = np.array(features)
            with open(output_file, 'wb') as f:
                np.save(f, np_feat)
            track_pbar.update(1)
            

if __name__ == "__main__":
    print("Running Centroids-ReID Feature Generation")
    parser = argparse.ArgumentParser()
    parser.add_argument('--tracklets_folder', help="Folder containing tracklet directories with images")
    parser.add_argument('--output_folder', help="Folder to store features in, one file per tracklet")
    args = parser.parse_args()

    #create if does not exist
    Path(args.output_folder).mkdir(parents=True, exist_ok=True)
    print(f'centroid_reid: main() tracklets_folder={args.tracklets_folder}, output_folder={args.output_folder}')
    generate_features(args.tracklets_folder, args.output_folder)



