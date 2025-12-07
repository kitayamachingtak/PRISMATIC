import pandas as pd
import os
import shutil
from tqdm import tqdm

#The dataset is a zip, so I can only download the full dataset then filter it
#Sorry that I haven't found a way to direct download images from hugging face

def copy_images_by_split(csv_path, source_folder, train_folder, test_folder):
    try:
        for folder in [train_folder, test_folder]:
            if not os.path.exists(folder):
                os.makedirs(folder)
        
        df = pd.read_csv(csv_path)
        train_filenames = df[df['split'] == 'train']['filename'].unique()
        test_filenames = df[df['split'] == 'test']['filename'].unique()
        source_files = set(os.listdir(source_folder))
        
        train_found = 0
        train_not_found = 0
        
        for filename in tqdm(train_filenames, desc="Train"):
            if filename in source_files:
                source_path = os.path.join(source_folder, filename)
                dest_path = os.path.join(train_folder, filename)
                shutil.copy2(source_path, dest_path)
                train_found += 1
            else:
                train_not_found += 1
        
        test_found = 0
        test_not_found = 0
        
        for filename in tqdm(test_filenames, desc="Test"):
            if filename in source_files:
                source_path = os.path.join(source_folder, filename)
                dest_path = os.path.join(test_folder, filename)
                shutil.copy2(source_path, dest_path)
                test_found += 1
            else:
                test_not_found += 1
        
        print(f"\nTrain: {train_found} copied, {train_not_found} not found")
        print(f"Test: {test_found} copied, {test_not_found} not found")
        
    except Exception as e:
        print(f"Error: {str(e)}")

csv_path = "full_dataset.csv"
source_folder = "path/to/source/images" #you should download from flickr 30k from hugging face and put it here
train_folder = "images/train"
test_folder = "images/test"

copy_images_by_split(csv_path, source_folder, train_folder, test_folder)