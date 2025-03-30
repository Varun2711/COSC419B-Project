## Setup

```shell
conda create -n open-mmlab python=3.8 pytorch=1.10 cudatoolkit=11.3 torchvision -c pytorch -y
conda activate open-mmlab
pip3 install openmim
git clone https://github.com/open-mmlab/mmocr.git
cd mmocr
mim install -e .
```

Please go to https://github.com/open-mmlab/mmocr for more details!

## Structure

+---data
|       DATASET_HERE
|       
+---model
|       TRAINED_MODEL_HERE
|       
\---script
    |   config.py
    |   config.txt
    |   dataset.py
    |   init_model.py
    |   test.py
    |   
    \---Dictionary
            digits_empty.txt (store label 0-9)

## Training
```commandline
python init_model.py
```

### Test
```commandline
python test.py
```

### Inference
```commandline
python inference.py
```