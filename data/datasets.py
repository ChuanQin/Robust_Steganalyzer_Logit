import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import cv2
from scipy import io
from glob import glob
import os

class ImageWithNameDataset(Dataset):

    def __init__(self, img_dir, indices = None, ref_dir = None, transform = None, repeat = 1):
        super(ImageWithNameDataset, self).__init__()

        self.img_dir = img_dir
        self.indices = indices
        self.ref_dir = ref_dir
        full_img_list = sorted(glob(self.img_dir + '/*'))
        if indices is not None:
            self.img_list = [full_img_list[i-1] for i in indices]
        elif ref_dir is not None:
            self.img_list = os.listdir(ref_dir)
        else:
            self.img_list = full_img_list.copy()
        if np.size(np.where(np.asarray(self.img_dir.split('/'))=='cover'))!=0 or \
        any([any(np.asarray(dir_str.split('_'))=='cover') for dir_str in self.img_dir.split('/')]):
            self.label_list = np.zeros(len(self.img_list))
        else:
            self.label_list = np.ones(len(self.img_list))
        self.len = len(self.label_list)
        self.repeat = repeat
        self.transform = transform

    def __getitem__(self, i):
        index = i % self.len
        label = np.array(self.label_list[index])
        if self.ref_dir is None:
            image_path = self.img_list[index]
        else:
            image_path = self.img_dir + '/' + self.img_list[index]
        img = self.transform(Image.open(image_path))
        return img, image_path.split('/')[-1], label

    def __len__(self):
        if self.repeat == None:
            data_len = 10000000
        else:
            data_len = self.len * self.repeat
        return data_len
