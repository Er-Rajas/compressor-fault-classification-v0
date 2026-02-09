import os,sys
os.environ['KAGGLEHUB_CACHE'] = r'D:\Project\Predictive maintaince\Data'

import kagglehub

# Download latest version
path = kagglehub.dataset_download("samoilovmikhail/simulated-refrigerator-fault-diagnosis-dataset")

print("Path to dataset files:", path)