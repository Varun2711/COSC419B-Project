echo y | conda create --name SoccerNet python=3.9
conda activate SoccerNet
pip install gdown
pip install tqdm
pip install SoccerNet
pip install pandas
pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
ehco y | apt-get install libgl1
python SetupSoccerNetDataset.py
python setup.py SoccerNet
