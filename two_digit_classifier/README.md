# Two Digit Classifier

This part of the code uses a CNN based simple architecture (ResNet18) to replace the Parseq architecture to reduce model complexity.

## Setup
```
> pip install -r requirements.txt
```

## Train the model

```
> python main.py train
```

## Test
```
> python main.py test
```

### Options
```
> python main.py --help
usage: main.py [-h] [--data_dir DATA_DIR] [--gt_file GT_FILE] [--batch_size BATCH_SIZE] [--gpu GPU] [--model_dir MODEL_DIR] [--saved_model SAVED_MODEL] [--arch ARCH] [--epochs EPOCHS]
               {train,test}

Jersey Number Recognition

positional arguments:
  {train,test}          Operation mode (train/test)

options:
  -h, --help            show this help message and exit
  --data_dir DATA_DIR   Override dataset directory
  --gt_file GT_FILE     Override ground truth path
  --batch_size BATCH_SIZE
                        Batch size for data loading
  --gpu GPU             GPU device number to use
  --model_dir MODEL_DIR
                        Override model path
  --saved_model SAVED_MODEL
                        Model filename (test)
  --arch ARCH           Model architecture (train)
  --epochs EPOCHS       Number of training epochs
```
