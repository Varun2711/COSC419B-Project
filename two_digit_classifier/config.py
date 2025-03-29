import argparse
import os
from torch.cuda import is_available as cuda_available
from torch.mps import is_available as mps_available
from torch import device as set_torch_device


def update_config(key, value):
    with open("config.txt", "r+") as f:
        lines = f.readlines()
        f.seek(0)
        for line in lines:
            if line.startswith(key):
                f.write(f"{key} = {value}\n")
            else:
                f.write(line)
        f.truncate()


def get_config(require_mode=True, default_mode=None):
    """Unified configuration loader that combines:
    - Config file loading
    - Argument parsing
    - Device setup
    - Path resolution
    Returns dictionary with all settings"""

    # Load configuration from file
    config = {}
    try:
        with open("config.txt", "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    key, value = line.split("=", 1)
                    config[key.strip()] = value.strip()
    except FileNotFoundError:
        raise SystemExit("Error: Missing config.txt file")

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Jersey Number Recognition")
    if require_mode:
        parser.add_argument(
            "mode", choices=["train", "test"], help="Operation mode (train/test)"
        )

    # Add common parameters
    parser.add_argument("--data_dir", help="Override dataset directory")
    parser.add_argument("--gt_file", help="Override ground truth path")
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for data loading"
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU device number to use")

    parser.add_argument("--model_dir", help="Override model path")
    parser.add_argument("--saved_model", help="Model filename (test)")

    parser.add_argument("--arch", default="resnet18", help="Model architecture (train)")
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs"
    )

    args = parser.parse_args()

    valid_arch_list = ["resnet18", "resnet34"]

    # Determine operation mode
    mode = args.mode if require_mode else default_mode
    if not mode:
        raise ValueError("Mode must be specified either through arguments or default")

    # Add model type validation
    if args.mode == "train" and args.arch and args.arch not in valid_arch_list:
        raise ValueError("Unsupported model architecture")

    # Configure compute device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    if cuda_available():
        device = set_torch_device("cuda")  # CUDA
    elif mps_available():
        device = set_torch_device("mps")  # macOS MPS (Metal Performance Shaders)
    else:
        device = set_torch_device("cpu")

    # Resolve file paths
    resolved = {
        "model_dir": config.get("model_dir", "models/"),
        "saved_model": args.saved_model
        or config.get("default_model", "resnet18_bs32_epoch10.pth"),
        "data_dir": args.data_dir or config.get(f"{mode}_data_dir"),
        "gt_file": args.gt_file or config.get(f"{mode}_gt_file"),
    }

    # Extract model architecture from filename
    test_model_arch = resolved["saved_model"][: resolved["saved_model"].find("_")]
    if args.mode == "test":
        if test_model_arch in valid_arch_list:
            args.arch = test_model_arch
        else:
            print(
                f"Cannot parse architecture from file name {resolved['saved_model']}, will use {args.arch}."
            )

    # Validate critical paths
    for key in ["data_dir", "gt_file"]:
        if not resolved[key]:
            raise ValueError(
                f"Missing required {key}. Provide via argument or config.txt"
            )

    return {
        "mode": mode,
        "device": device,
        "batch_size": args.batch_size,
        "model_arch": args.arch,
        "epochs": args.epochs,
        **resolved,
    }
