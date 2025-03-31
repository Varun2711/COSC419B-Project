import os
import zipfile
import glob
from tqdm import tqdm
from SoccerNet.Downloader import SoccerNetDownloader as SNdl


def unzip_with_progress(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        file_list = zip_ref.namelist()
        total_files = len(file_list)

        if total_files == 0:
            print(f"⚠️ {zip_path} is empty, skip it...")
            return
        for file in tqdm(file_list, desc=f"Extracting {os.path.basename(zip_path)}", unit="file"):
            zip_ref.extract(file, extract_to)


def batch_unzip(directory):
    zip_files = glob.glob(os.path.join(directory, "*.zip"))  # 获取所有 ZIP 文件列表
    if not zip_files:
        print("❌ no zip files found")
        return

    print(f"📂 In {directory}, found {len(zip_files)} zip files, unzipping now...\n")

    for zip_file in zip_files:
        # extract_folder = os.path.join(directory, os.path.splitext(os.path.basename(zip_file))[0])
        extract_folder = directory

        os.makedirs(extract_folder, exist_ok=True)
        unzip_with_progress(zip_file, extract_folder)

    print("\n✅ all files unzipped！")

root_dir = os.getcwd()
mySNdl = SNdl(LocalDirectory="data")
mySNdl.downloadDataTask(task="jersey-2023", split=["train", "test", "challenge"])

dataset_path = os.path.join(root_dir, 'data/jersey-2023')
if os.path.exists(dataset_path):
    dataset_path_old = dataset_path
    dataset_path = os.path.join(root_dir, 'data/SoccerNet')
    os.rename(dataset_path_old, dataset_path)
    print(f'Rename {dataset_path_old} to {dataset_path}')

    # unzip dataset
    batch_unzip(dataset_path)