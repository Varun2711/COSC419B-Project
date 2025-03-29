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



# Based on this repo: https://github.com/mikwieczorek/centroids-reid
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

def generate_features(input_folder, output_folder, model_version='res50_market', b_overwrite_cache=False):
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

    # print("Loading from " + MODEL_FILE)
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
    for track in tqdm(tracks):
        features = []
        track_path = os.path.join(input_folder, track)
        if not os.path.isdir(track_path):
            continue
        output_file = os.path.join(output_folder, f"{track}_features.npy")
        if os.path.exists(output_file) and not b_overwrite_cache:
            continue
        images = os.listdir(track_path)
        for img_path in images:
            if img_path.endswith('.DS_Store'):
                continue
            img = cv2.imread(os.path.join(track_path, img_path))
            input_img = Image.fromarray(img)
            input_img = torch.stack([val_transforms(input_img)])
            with torch.no_grad():
                _, global_feat = model.backbone(input_img.cuda() if use_cuda else input_img)
                global_feat = model.bn(global_feat)
            features.append(global_feat.cpu().numpy().reshape(-1,))

        np_feat = np.array(features)
        with open(output_file, 'wb') as f:
            # print(f'generate_features() Writing {output_file}.')
            np.save(f, np_feat)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tracklets_folder', help="Folder containing tracklet directories with images")
    parser.add_argument('--output_folder', help="Folder to store features in, one file per tracklet")
    args = parser.parse_args()

    #create if does not exist
    Path(args.output_folder).mkdir(parents=True, exist_ok=True)
    print(f'centroid_reid: main() tracklets_folder={args.tracklets_folder}, output_folder={args.output_folder}')
    generate_features(args.tracklets_folder, args.output_folder)



