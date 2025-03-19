import argparse
import os
from torch.cuda import is_available as cuda_available
from torch.mps import is_available as mps_available
from torch import device as set_torch_device


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
    parser.add_argument("--model_path", help="Override model path")

    args = parser.parse_args()

    # Determine operation mode
    mode = args.mode if require_mode else default_mode
    if not mode:
        raise ValueError("Mode must be specified either through arguments or default")

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
        "data_dir": args.data_dir or config.get(f"{mode}_data_dir"),
        "gt_file": args.gt_file or config.get(f"{mode}_gt_file"),
        "model_path": args.model_path or config.get("model_path"),
    }

    # Validate critical paths
    for key in ["data_dir", "gt_file", "model_path"]:
        if not resolved[key]:
            raise ValueError(
                f"Missing required {key}. Provide via argument or config.txt"
            )

    return {"mode": mode, "device": device, "batch_size": args.batch_size, **resolved}
