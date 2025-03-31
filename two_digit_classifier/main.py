from config import get_config


def main():
    cfg = get_config()  # require_mode=True by default

    if cfg["mode"] == "train":
        from train import train_model

        train_model(cfg)
    elif cfg["mode"] == "test":
        from test import test_model

        test_model(cfg)
    else:
        from inference import run_inference

        run_inference(cfg)


if __name__ == "__main__":
    main()
