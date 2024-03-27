"""
    Data-Loader Assignment, Deep Learning

    Submitted by:
        Name:    Arhum Ahmed
        Roll No: 2020-EE-123
        Section: B

"""

import os
import cv2
import numpy as np
import random

def load_images(data_dir):
    """
        This function loads the dataset.
        data_dir => Directory of the folder to extract data from
        data_set => Returns a list with tuple elements (img_array, label)
    """
    data_set = []
    list_classes = os.listdir(data_dir)
    for i in list_classes:
        sub_dir   = os.path.join(data_dir, i)
        sub_class = os.listdir(sub_dir)
        for img in sub_class:
            img_path = os.path.join(sub_dir, img)
            img_array = np.array(cv2.imread(img_path))
            data_set.append((img_array, i))
    return data_set


class DataLoader():
    def __init__(self, dataset, batch_size, shuffle=True):
        self.batch_size = batch_size
        self.dataset    = dataset
        self.shuffle    = shuffle
    
    def __next__(self):
        batch_out  = []
        label_out  = []
        sample_out = []

        if self.shuffle:
            random.shuffle(self.dataset)

        for i in range(self.batch_size):
            sample, label = self.dataset[i]
            sample_out.append(sample)
            label_out.append(label)
        batch_out.append(tuple(sample_out))
        batch_out.append(tuple(label_out))
        return batch_out




data_set_train = load_images('train')

train_data = DataLoader(data_set_train, batch_size=100, shuffle=True)

print(next(train_data))