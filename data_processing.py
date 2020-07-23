import pandas as pd
import numpy as np
from torch.utils.data import Dataset,DataLoader
import os
import cv2
import matplotlib.pyplot as plt
class FacialKeyPointDataProccessing(Dataset):

    def __init__(self, csv_file, image_directory, transformation=None):
        self.key_points = pd.read_csv(csv_file)
        self.image_folder = image_directory
        self.transform = transformation

    def __len__(self):
        return len(self.key_points)

    def __getitem__(self, item_index):
        image_directory = os.path.join(self.image_folder, self.key_points.iloc[item_index,0])
        image = plt.imread(image_directory, cv2.COLOR_BGR2RGB)
        image = image[:,:,0:3]
        key_points = self.key_points.iloc[item_index,1:].to_numpy()
        key_points = key_points.astype("float").reshape(-1,2)
        data = {"keypoints": key_points, "image":image}

        if(self.transform is not None):
            data = self.transform(data)

        return data